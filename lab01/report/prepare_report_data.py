"""Collect report evidence without changing the experiment code or checkpoints."""
import contextlib
import csv
import hashlib
import io
import json
import platform
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import sklearn
import wandb
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
FIG = OUT / 'figures'
sys.path.insert(0, str(ROOT))
from model import MLP
from data import get_test_loader

NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
         'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
CN = ['T恤/上衣', '裤子', '套头衫', '连衣裙', '外套', '凉鞋', '衬衫', '运动鞋', '包', '短靴']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    sys.stdout.reconfigure(encoding='utf-8')
    torch.set_num_threads(4)
    plan = json.loads((ROOT / 'results/_batch/night01/plan.json').read_text(encoding='utf-8'))
    runs = {}
    sources = []
    for job in plan:
        directory = ROOT / 'results' / ('night01_' + job['label'])
        summary = json.loads((directory / 'test_summary.json').read_text(encoding='utf-8'))
        def rows(filename):
            with (directory / filename).open(encoding='utf-8') as file:
                return [{k: int(v) if k == 'epoch' else float(v) for k, v in row.items()}
                        for row in csv.DictReader(file)]
        history, tests = rows('history.csv'), rows('test_history.csv')
        assert summary['train_history_sha256'] == sha(directory / 'history.csv')
        assert len(history) == len(tests) == job['epochs']
        assert summary['complete_test_curve'] and not summary['missing_epochs']
        best = min(history, key=lambda row: row['val_error'])
        assert best['epoch'] == summary['best_epoch']
        assert tests[best['epoch'] - 1]['test_error'] == summary['best_test_error']
        runs[job['label']] = {'summary': summary, 'history': history, 'tests': tests}
        for filename in ('history.csv', 'test_history.csv', 'test_summary.json'):
            sources.append({'path': str(directory / filename), 'sha256': sha(directory / filename)})

    # The model is chosen by validation error before this diagnostic pass.
    chosen_label = min(runs, key=lambda key: runs[key]['summary']['best_val_error'])
    assert chosen_label == 'large_bn'
    selected_run = runs[chosen_label]
    checkpoint_path = ROOT / 'results/night01_large_bn/best.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP(784, config['hidden_dims'], 10, config['activation'],
                config['dropout'], config['batch_norm']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    loader = get_test_loader(ROOT / 'data', 64)
    predictions, truths, confidences = [], [], []
    total_loss = 0.
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            probability = logits.softmax(dim=1)
            confidence, prediction = probability.max(dim=1)
            predictions.extend(prediction.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())
            truths.extend(labels.tolist())
            total_loss += nn.functional.cross_entropy(logits, labels.to(device), reduction='sum').item()
    predictions, truths = np.array(predictions), np.array(truths)
    cm = np.bincount(truths * 10 + predictions, minlength=100).reshape(10, 10)
    assert np.all(cm.sum(axis=1) == 1000)
    errors = int((predictions != truths).sum())
    assert errors == 1003
    assert errors / 10000 == selected_run['summary']['best_test_error']
    assert abs(total_loss / 10000 - selected_run['summary']['best_test_loss']) < 1e-6
    per_class = [{'class_id': i, 'class_name': NAMES[i], 'support': int(cm[i].sum()),
                  'correct': int(cm[i, i]), 'recall': float(cm[i, i] / cm[i].sum())} for i in range(10)]
    pairs = sorted([(int(cm[i, j]), i, j) for i in range(10) for j in range(10) if i != j], reverse=True)
    examples = []
    for count, true, pred in pairs[:5]:
        index = int(np.flatnonzero((truths == true) & (predictions == pred))[0])
        examples.append({'index': index, 'true': true, 'pred': pred, 'pair_count': count,
                         'confidence': float(confidences[index]),
                         'pixels': loader.dataset.data[index].tolist()})
    # One deterministic example per class, from the original training split.
    training = torchvision.datasets.FashionMNIST(ROOT / 'data', train=True, download=False)
    samples = []
    for label in range(10):
        index = int(torch.where(training.targets == label)[0][0])
        samples.append({'class_id': label, 'index': index, 'pixels': training.data[index].tolist()})
    with (OUT / 'bn_test_predictions.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['test_index', 'true', 'predicted', 'confidence'])
        writer.writerows(zip(range(10000), truths.tolist(), predictions.tolist(), confidences))
    with (OUT / 'bn_per_class.csv').open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=per_class[0].keys())
        writer.writeheader()
        writer.writerows(per_class)
    np.savetxt(OUT / 'bn_confusion_matrix.csv', cm, fmt='%d', delimiter=',')

    # Execute the user's XOR.py unchanged; intercept the already-computed MSE.
    xor_source = ROOT / 'XOR.py'
    history = []
    original_forward = nn.MSELoss.forward
    def capture_loss(self, inputs, targets):
        loss = original_forward(self, inputs, targets)
        history.append(float(loss.detach()))
        return loss
    stdout = io.StringIO()
    with patch.object(nn.MSELoss, 'forward', capture_loss), contextlib.redirect_stdout(stdout):
        namespace = runpy.run_path(str(xor_source), run_name='__main__')
    assert len(history) == 5000
    xor_model = namespace['model'].eval()
    with torch.no_grad():
        values = xor_model(namespace['x']).reshape(-1)
        final_mse = float(nn.functional.mse_loss(values, namespace['y'].reshape(-1)))
        gx = np.linspace(-0.25, 1.25, 41)
        xx, yy = np.meshgrid(gx, gx)
        grid = torch.tensor(np.column_stack([xx.ravel(), yy.ravel()]), dtype=torch.float32)
        zz = xor_model(grid).reshape(xx.shape).numpy()
    history.append(final_mse)
    classes = (values > 0.5).int().tolist()
    assert classes == [0, 1, 1, 0]
    with (OUT / 'xor_history.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['completed_updates', 'mse'])
        writer.writerows(enumerate(history))
    (OUT / 'xor_stdout.txt').write_text(stdout.getvalue(), encoding='utf-8')

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
    for ax, item in zip(axes.ravel(), samples):
        ax.imshow(item['pixels'], cmap='gray', vmin=0, vmax=255)
        ax.set_title(f"{item['class_id']}: {NAMES[item['class_id']]}", fontsize=10)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(FIG / 'fashion_samples.png', dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1.1, 1]})
    axes[0].imshow(cm, cmap='Blues', vmin=0, vmax=1000)
    for i in range(10):
        for j in range(10):
            axes[0].text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=7,
                         color='white' if cm[i,j]>500 else 'black')
    axes[0].set(xticks=range(10), yticks=range(10), xlabel='Predicted class', ylabel='True class',
                title='BN, epoch 52: test confusion counts')
    axes[1].barh(NAMES, [row['recall']*100 for row in per_class], color='#3368A5')
    axes[1].invert_yaxis()
    axes[1].set(xlim=(0, 100), xlabel='Recall (%)', title='Per-class recall (1,000 samples each)')
    fig.tight_layout()
    fig.savefig(FIG / 'bn_error_analysis.png', dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(1, 5, figsize=(12, 3))
    for ax, item in zip(axes, examples):
        ax.imshow(item['pixels'], cmap='gray', vmin=0, vmax=255)
        ax.set_title(f"#{item['index']}\n{NAMES[item['true']]} -> {NAMES[item['pred']]}", fontsize=9)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(FIG / 'bn_error_examples.png', dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].semilogy(range(5001), history, color='#3368A5')
    axes[0].set(xlabel='Completed SGD updates', ylabel='MSE (log scale)', title='XOR fitting: seed=1, lr=0.1')
    axes[0].grid(alpha=.25)
    axes[1].contourf(xx, yy, zz>.5, levels=[-.5,.5,1.5], colors=['#e0eaf4','#f2dfbe'])
    contour=axes[1].contour(xx, yy, zz, levels=[.5], colors=['#333333'], linewidths=1.5)
    segments=[segment.tolist() for segment in contour.allsegs[0] if len(segment)>1]
    axes[1].scatter([0,1],[0,1],c='#3368A5',marker='o',s=80,label='Class 0')
    axes[1].scatter([0,1],[1,0],c='#B77D25',marker='s',s=80,label='Class 1')
    axes[1].set(xlabel='x1',ylabel='x2',title='Threshold = 0.5; grid is illustrative')
    axes[1].legend(frameon=False, loc='upper center', ncols=2)
    axes[1].set_aspect('equal')
    fig.tight_layout()
    fig.savefig(FIG / 'xor_learning_boundary.png', dpi=200)
    plt.close(fig)

    evidence = {'runs': runs, 'selected_run': chosen_label,
                'classes': [{'id':i,'english':NAMES[i],'chinese':CN[i]} for i in range(10)],
                'environment': {'python':platform.python_version(),'torch':torch.__version__,
                                'torchvision':torchvision.__version__,'numpy':np.__version__,
                                'sklearn':sklearn.__version__,'wandb':wandb.__version__,
                                'cuda':torch.version.cuda,'gpu':'CUDA GPU' if torch.cuda.is_available() else 'CPU'},
                'samples':samples, 'confusion_matrix':cm.tolist(), 'per_class':per_class,
                'test_errors':errors,'top_confusions':[{'count':count,'true':i,'pred':j} for count,i,j in pairs[:10]],
                'error_examples':examples,
                'xor': {'seed':1,'updates':5000,'learning_rate':.1,'parameters':sum(p.numel() for p in xor_model.parameters()),
                        'initial_mse':history[0],'final_mse':final_mse,'predictions':values.tolist(),'classes':classes,
                        'history':history,'grid_axis':gx.tolist(),'grid_output':zz.tolist(),'boundary_segments':segments},
                'sources':sources+[{'path':str(checkpoint_path),'sha256':sha(checkpoint_path)},
                                    {'path':str(xor_source),'sha256':sha(xor_source)}]}
    (OUT/'report_evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'selected_run':chosen_label,'test_errors':errors,'per_class':per_class,
                      'top_confusions':evidence['top_confusions'][:5],
                      'xor':{k:v for k,v in evidence['xor'].items() if k not in ('history','grid_axis','grid_output','boundary_segments')},
                      'environment':evidence['environment']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
