"""Build a self-contained LaTeX report with embedded vector figures and data."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
E = json.loads((ROOT / 'report_evidence.json').read_text(encoding='utf-8'))
RUNS = E['runs']
KEYS = list(RUNS)
IDS = {key: ('A' + str(i) if i < 7 else 'B' + str(i - 7)) for i, key in enumerate(KEYS)}
COLORS = ['PBlue', 'PGold', 'POrange', 'POlive', 'PPink']
STYLES = ['solid', 'dashed', 'dashdotted', 'dotted', 'solid']
NAMES = {
    'baseline': '基线', 'narrow': '窄网络', 'wide': '宽网络', 'deep': '深网络',
    'tanh': 'Tanh', 'sigmoid': 'Sigmoid', 'mse': '概率 MSE',
    'large_plain': '无正则化', 'large_l2_1e-4': r'L2，$\lambda=10^{-4}$',
    'large_l2_1e-3': r'L2，$\lambda=10^{-3}$', 'large_dropout': r'Dropout，$p=0.2$', 'large_bn': 'BatchNorm',
}


def esc(value):
    return str(value).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')


def pct(value):
    return f'{100 * value:.2f}' + r'\%'


def table_row(*values):
    return ' & '.join(map(str, values)) + r' \\' + '\n'


def best(key, budget=None):
    run = RUNS[key]
    rows = run['history'] if budget is None else run['history'][:budget]
    row = min(rows, key=lambda r: r['val_error'])
    return row, run['tests'][row['epoch'] - 1]


def number(value):
    return f'{value:.8g}'


def coordinates(xs, ys):
    return ' '.join(f'({number(x)},{number(y)})' for x, y in zip(xs, ys))


def plot(series, title='', ylabel='', options='', height='5.2cm', legend_columns=2, best_epoch=None):
    parts = [r'\begin{tikzpicture}\begin{axis}[reportaxis,'
             + f'height={height},title={{{title}}},xlabel={{Epoch}},ylabel={{{ylabel}}},'
             + f'legend columns={legend_columns},' + options + ']\n']
    for i, item in enumerate(series):
        color = item.get('color', COLORS[i % 5])
        style = item.get('style', STYLES[i % 5])
        parts.append(r'\addplot[' + color + ',' + style + ',line width=0.8pt,no marks] coordinates {'
                     + coordinates(item['x'], item['y']) + '};\n')
        parts.append(r'\addlegendentry{' + item['label'] + '}\n')
    if best_epoch is not None:
        parts.append(r'\addplot[PGray,densely dotted,forget plot] coordinates {'
                     + f'({best_epoch},0)({best_epoch},18)' + '};\n')
    parts.append(r'\end{axis}\end{tikzpicture}')
    return ''.join(parts)


def lines(keys, field, budget=None, labels=None, source='history'):
    result = []
    for i, key in enumerate(keys):
        rows = RUNS[key][source]
        if budget:
            rows = rows[:budget]
        result.append({'label': labels[i] if labels else IDS[key],
                       'x': [r['epoch'] for r in rows],
                       'y': [r[field] * (100 if 'error' in field else 1) for r in rows]})
    return result


def pair(left, right):
    return (r'\begin{minipage}[t]{0.49\textwidth}\vspace{0pt}\centering' + left
            + r'\end{minipage}\hfill\begin{minipage}[t]{0.49\textwidth}\vspace{0pt}\centering'
            + right + r'\end{minipage}')


def pixel_image(pixels, scale=.073):
    parts = [f'\\begin{{tikzpicture}}[x={scale}cm,y={scale}cm]\n',
             r'\fill[black] (0,0) rectangle (28,28);' + '\n']
    # Merge equal neighboring pixels to retain all gray levels with fewer paths.
    for row_index, row in enumerate(pixels):
        column = 0
        while column < 28:
            end = column + 1
            while end < 28 and row[end] == row[column]:
                end += 1
            if row[column] != 0:
                darkness = 100 * (1 - row[column] / 255)
                parts.append(f'\\fill[black!{darkness:.4f}] ({column},{27-row_index}) rectangle ({end},{28-row_index});\n')
            column = end
    parts.append(r'\end{tikzpicture}')
    return ''.join(parts)


def image_grid(items, is_error=False):
    parts = []
    for i, item in enumerate(items):
        parts.append(r'\begin{minipage}[t]{0.19\textwidth}\centering')
        parts.append(pixel_image(item['pixels']))
        if is_error:
            actual = E['classes'][item['true']]['chinese']
            predicted = E['classes'][item['pred']]['chinese']
            parts.append(r'\par{\scriptsize ' + f"\#{item['index']}" + r'\par '
                         + actual + r'$\to$' + predicted + '}')
        else:
            label = item['class_id']
            parts.append(r'\par{\small ' + str(label) + '：' + E['classes'][label]['chinese'] + '}')
        parts.append(r'\end{minipage}')
        if (i + 1) % 5 == 0 and i + 1 < len(items):
            parts.append(r'\par\vspace{0.5em}')
        elif (i + 1) % 5:
            parts.append(r'\hfill')
    return ''.join(parts)


def confusion():
    parts = [r'\begin{tikzpicture}[x=0.59cm,y=0.59cm,font=\scriptsize]' + '\n']
    for row in range(10):
        for col in range(10):
            value = E['confusion_matrix'][row][col]
            parts.append(f'\\fill[PBlue!{value / 10:.1f}!white] ({col},{9-row}) rectangle ({col+1},{10-row});\n')
            color = 'white' if value > 500 else 'black'
            parts.append(f'\\node[text={color}] at ({col+.5},{9.5-row}) {{{value}}};\n')
    for i in range(10):
        parts.append(f'\\node at ({i+.5},-.35) {{{i}}};\\node at (-.35,{9.5-i}) {{{i}}};\n')
    parts.extend([r'\draw[black!40] (0,0) rectangle (10,10);',
                  r'\node at (5,-1.05) {预测类别};',
                  r'\node[rotate=90] at (-1.05,5) {真实类别};',
                  r'\end{tikzpicture}'])
    recall = [r'\begin{tikzpicture}\begin{axis}[reportaxis,width=7cm,height=6.6cm,xbar,bar width=9pt,',
              r'xmin=0,xmax=110,ymin=-.7,ymax=9.7,y dir=reverse,ytick={0,1,2,3,4,5,6,7,8,9},',
              'yticklabels={' + ','.join(item['chinese'] for item in E['classes']) + '},',
              r'xlabel={召回率（\%）},xtick={0,25,50,75,100},nodes near coords,',
              r'every node near coord/.append style={font=\scriptsize,/pgf/number format/fixed,/pgf/number format/precision=1},',
              r'grid=none,axis x line*=bottom,axis y line*=left]',
              r'\addplot[fill=PBlue,draw=none] coordinates {'
              + coordinates([x['recall'] * 100 for x in E['per_class']], list(range(10))) + '};',
              r'\end{axis}\end{tikzpicture}']
    return pair(''.join(parts), ''.join(recall))


def xor_plot():
    x = E['xor']
    steps = list(range(0, 5001, 10))
    curve = plot([{'label': '四样本 MSE', 'x': steps, 'y': [x['history'][i] for i in steps]}],
                 'XOR 训练损失', 'MSE',
                 options=r'xlabel={SGD 更新次数},ymode=log,xmin=0,xmax=5000,ymin=1e-14,ymax=1,xtick={0,1000,2000,3000,4000,5000}',
                 height='5.6cm')
    axis, values = x['grid_axis'], x['grid_output']
    # The background is a piecewise constant depiction of the evaluated grid.
    parts = [r'\begin{tikzpicture}[x=3.25cm,y=3.25cm,font=\scriptsize]',
             r'\fill[PBlue!12] (-.25,-.25) rectangle (1.25,1.25);']
    for row in range(40):
        col = 0
        positive = [(values[row][j] + values[row+1][j] + values[row][j+1] + values[row+1][j+1]) / 4 > .5 for j in range(40)]
        while col < 40:
            end = col + 1
            while end < 40 and positive[end] == positive[col]:
                end += 1
            if positive[col]:
                parts.append(f'\\fill[PGold!22] ({axis[col]:.5f},{axis[row]:.5f}) rectangle ({axis[end]:.5f},{axis[row+1]:.5f});')
            col = end
    for segment in x['boundary_segments']:
        parts.append(r'\draw[black!75,line width=.9pt] ' + ' -- '.join(f'({p[0]:.5f},{p[1]:.5f})' for p in segment) + ';')
    parts.extend([r'\fill[PBlue] (0,0) circle (2pt);\fill[PBlue] (1,1) circle (2pt);',
                  r'\fill[PGold] (-.025,.975) rectangle (.025,1.025);\fill[PGold] (.975,-.025) rectangle (1.025,.025);',
                  r'\node[below left] at (0,0) {0};\node[above right] at (1,1) {0};',
                  r'\node[above left] at (0,1) {1};\node[below right] at (1,0) {1};',
                  r'\draw[black!50] (-.25,-.25) rectangle (1.25,1.25);',
                  r'\node at (.5,1.38) {预测区域与训练点};',
                  r'\node at (.5,-.45) {$x_1$};\node[rotate=90] at (-.60,.5) {$x_2$};',
                  r'\foreach \t in {0,.5,1}{\draw (\t,-.25)--(\t,-.28) node[below]{\t};\draw (-.25,\t)--(-.28,\t) node[left]{\t};}',
                  r'\node[align=center] at (.5,-.70) {浅蓝：预测 0\quad 浅金：预测 1\\黑线：输出为 0.5 的等值线};',
                  r'\end{tikzpicture}'])
    return pair(curve, ''.join(parts))


def question_content():
    """Figures and comparison tables placed directly under each PPT question."""
    replacements = {}

    def triple(keys, labels, budget):
        panels = []
        ticks = '1,5,10,15,20' if budget == 20 else '1,20,40,60'
        error_max = max(
            r[field] * 100 for key in keys
            for source, field in [('history', 'train_error'), ('tests', 'test_error')]
            for r in RUNS[key][source][:budget]
        )
        error_top = max(18, int(error_max / 2 + 1) * 2)
        for field, title, ylabel, source in [
            ('train_loss', '训练损失', 'Loss', 'history'),
            ('train_error', '训练误差', r'误差（\%）', 'history'),
            ('test_error', '测试误差', r'误差（\%）', 'tests'),
        ]:
            options = f'width=5.45cm,xmin=1,xmax={budget},ymin=0,xtick={{{ticks}}}'
            if 'error' in field:
                options += f',ymax={error_top}'
            content = plot(lines(keys, field, budget, labels, source), title, ylabel,
                           options=options, height='4.8cm')
            content = re.sub(r'\\addlegendentry\{[^\n]*\}\n', '', content)
            panels.append(r'\begin{minipage}[t]{0.325\textwidth}\vspace{0pt}\centering'
                          + content + r'\end{minipage}')
        legend = []
        for i, label in enumerate(labels):
            legend.append(r'\tikz[baseline=-.5ex]\draw[' + COLORS[i] + ',' + STYLES[i]
                          + r',line width=.8pt] (0,0)--(.55,0);\ ' + label)
        return r'\hfill'.join(panels) + r'\par\vspace{0.5em}{\small ' + r'\quad '.join(legend) + '}'

    structure = ['narrow', 'baseline', 'wide', 'deep']
    widths = lambda key: '--'.join(map(str, RUNS[key]['summary']['config']['hidden_dims']))
    replacements['BASELINE_TRIPLE'] = triple(['baseline'], ['256--128 / ReLU / CE'], 20)
    replacements['STRUCTURE_TRIPLE'] = triple(structure, [widths(k) for k in structure], 20)
    replacements['BASIC_STRUCTURE_ROWS'] = ''.join(table_row(
        widths(key), f"{RUNS[key]['summary']['num_params']:,}", best(key)[0]['epoch'],
        pct(best(key)[0]['val_error']), pct(best(key)[1]['test_error'])) for key in structure)
    replacements['ACTIVATION_TRIPLE'] = triple(['baseline', 'tanh', 'sigmoid'], ['ReLU', 'Tanh', 'Sigmoid'], 20)
    opts = 'xmin=1,xmax=20,ymin=0,xtick={1,5,10,15,20}'
    replacements['LOSS_FOUR_PANELS'] = pair(
        plot(lines(['baseline'], 'train_loss', labels=['CE']), '交叉熵训练损失', 'CE', options=opts, height='4.4cm'),
        plot(lines(['mse'], 'train_loss', labels=['MSE']), '概率 MSE 训练损失', 'MSE', options=opts, height='4.4cm'))
    replacements['LOSS_FOUR_PANELS'] += r'\par\vspace{1.2em}' + pair(
        plot(lines(['baseline', 'mse'], 'train_error', labels=['CE', 'MSE']), '训练误差', r'误差（\%）',
             options=opts + ',ymax=18', height='4.4cm'),
        plot(lines(['baseline', 'mse'], 'test_error', labels=['CE', 'MSE'], source='tests'), '测试误差', r'误差（\%）',
             options=opts + ',ymax=18', height='4.4cm'))
    replacements['COMPLEXITY_ROWS'] = ''.join(table_row(widths(key), f"{RUNS[key]['summary']['num_params']:,}",
        pct(RUNS[key]['history'][19]['train_error']), pct(RUNS[key]['tests'][19]['test_error']))
        for key in structure + ['large_plain'])
    plain = RUNS['large_plain']
    errors = [{'label': label, 'x': [r['epoch'] for r in plain[source]], 'y': [100*r[field] for r in plain[source]]}
              for label, source, field in [('训练', 'history', 'train_error'), ('验证', 'history', 'val_error'), ('测试', 'tests', 'test_error')]]
    replacements['OVERFIT_QUESTION'] = pair(
        plot(lines(['large_plain'], 'train_loss', labels=['训练']) + lines(['large_plain'], 'val_loss', labels=['验证']),
             '训练与验证损失', 'CE', options='xmin=1,xmax=60,ymin=0,xtick={1,20,40,60}'),
        plot(errors, '分类误差与泛化差距', r'误差（\%）', legend_columns=3,
             options='xmin=1,xmax=60,ymin=0,ymax=18,xtick={1,20,40,60}'))
    groups = {
        'L2': (['large_plain', 'large_l2_1e-4', 'large_l2_1e-3'],
               [r'$\lambda=0$', r'$\lambda=10^{-4}$', r'$\lambda=10^{-3}$']),
        'DROPOUT': (['large_plain', 'large_dropout'], ['无 Dropout', r'Dropout ($p=0.2$)']),
        'BN': (['large_plain', 'large_bn'], ['无 BN', 'Batch Normalization']),
    }
    for prefix, (keys, labels) in groups.items():
        replacements[prefix + '_TRIPLE'] = triple(keys, labels, 60)
        replacements[prefix + '_ROWS'] = ''.join(table_row(label,
            pct(RUNS[key]['history'][-1]['train_error']), pct(RUNS[key]['tests'][-1]['test_error']),
            pct(best(key)[1]['test_error'])) for key, label in zip(keys, labels))
    return replacements


def main():
    replacements = {key.upper(): esc(value) for key, value in E['environment'].items()}
    replacements['SAMPLES'] = image_grid(E['samples'])
    designs = []
    for key, run in RUNS.items():
        s = run['summary']
        widths = '--'.join(map(str, s['config']['hidden_dims']))
        desc = {'baseline': '默认基线', 'narrow': '减小宽度', 'wide': '增大宽度', 'deep': '增加一个隐藏层',
                'tanh': '激活改为 Tanh', 'sigmoid': '激活改为 Sigmoid', 'mse': '损失改为概率 MSE',
                'large_plain': '大模型，不启用正则化', 'large_l2_1e-4': r'$\lambda=10^{-4}$',
                'large_l2_1e-3': r'$\lambda=10^{-3}$', 'large_dropout': r'$p=0.2$', 'large_bn': '启用 BatchNorm'}[key]
        designs.append(table_row(IDS[key], widths, desc, s['completed_epochs'], f"{s['num_params']:,}"))
    replacements['DESIGN_ROWS'] = ''.join(designs)

    structure = ['narrow','baseline','wide','deep','large_plain']
    replacements['STRUCTURE_ROWS'] = ''.join(table_row(
        IDS[key], '--'.join(map(str, RUNS[key]['summary']['config']['hidden_dims'])),
        f"{RUNS[key]['summary']['num_params']:,}", best(key,20)[0]['epoch'],
        pct(best(key,20)[0]['val_error']), pct(best(key,20)[1]['test_error'])) for key in structure)
    left = plot(lines(structure,'val_error',20), '验证误差随训练变化', '分类误差（\%）',
                options='xmin=1,xmax=20,ymin=8,ymax=17,xtick={1,5,10,15,20}',legend_columns=3)
    xs = [RUNS[key]['summary']['num_params'] for key in structure]
    scatter = [r'\begin{tikzpicture}\begin{axis}[reportaxis,title={参数量与分类表现},xmode=log,',
               r'xlabel={可学习参数量（对数刻度）},ylabel={分类误差（\%）},xmin=40000,xmax=3500000,ymin=8,ymax=13,',
               r'xtick={100000,1000000},xticklabels={$10^5$,$10^6$}]']
    for field, color, mark, name in [('val_error','PBlue','*','验证'),('test_error','PGold','square*','测试')]:
        values = [best(key,20)[0 if field=='val_error' else 1][field]*100 for key in structure]
        scatter.append(r'\addplot[only marks,' + color + f',mark={mark},mark size=2pt] coordinates {{' + coordinates(xs,values) + '};')
        scatter.append(r'\addlegendentry{' + name + '}')
    for key, count in zip(structure, xs):
        val = best(key,20)[0]['val_error']*100
        position = 'below left' if key=='baseline' else 'above right'
        scatter.append(f'\\node[font=\\scriptsize,{position}] at (axis cs:{count},{val:.4f}) {{{IDS[key]}}};')
    scatter.append(r'\end{axis}\end{tikzpicture}')
    replacements['STRUCTURE_PLOTS'] = pair(left,''.join(scatter))

    activations=['baseline','tanh','sigmoid']
    replacements['ACTIVATION_ROWS']=''.join(table_row(label,best(key)[0]['epoch'],pct(best(key)[0]['val_error']),pct(best(key)[1]['test_error']))
                                           for key,label in zip(activations,['ReLU','Tanh','Sigmoid']))
    replacements['LOSS_ROWS']=''.join(table_row(label,best(key)[0]['epoch'],pct(best(key)[0]['val_error']),pct(best(key)[1]['test_error']))
                                     for key,label in zip(['baseline','mse'],['交叉熵 CE','Softmax + one-hot MSE']))
    replacements['ACTLOSS_PLOTS']=pair(
        plot(lines(activations,'val_error',labels=['ReLU','Tanh','Sigmoid']),'激活函数对比','分类误差（\%）',
             options='xmin=1,xmax=20,ymin=8,ymax=17,xtick={1,5,10,15,20}',legend_columns=3),
        plot(lines(['baseline','mse'],'val_error',labels=['CE','MSE']),'损失函数对比','分类误差（\%）',
             options='xmin=1,xmax=20,ymin=8,ymax=17,xtick={1,5,10,15,20}'))
    regularization=KEYS[7:]
    replacements['REG_ROWS']=''.join(table_row(IDS[key],NAMES[key],best(key)[0]['epoch'],pct(best(key)[0]['val_error']),pct(best(key)[1]['test_error'])) for key in regularization)
    replacements['FINAL_ROWS']=''.join(table_row(NAMES[key],pct(RUNS[key]['history'][-1]['train_error']),
        pct(RUNS[key]['history'][-1]['val_error']),pct(RUNS[key]['tests'][-1]['test_error']),
        f"{100*(RUNS[key]['history'][-1]['val_error']-RUNS[key]['history'][-1]['train_error']):.2f}") for key in regularization)
    replacements['REG_ERROR_PLOTS']=pair(
        plot(lines(regularization,'train_error'),'训练集分类误差','分类误差（\%）',
             options='xmin=1,xmax=60,ymin=0,ymax=18,xtick={1,10,20,30,40,50,60}',legend_columns=3),
        plot(lines(regularization,'test_error',source='tests'),'测试集分类误差','分类误差（\%）',
             options='xmin=1,xmax=60,ymin=0,ymax=18,xtick={1,10,20,30,40,50,60}',legend_columns=3))
    plain=RUNS['large_plain']
    errors=[{'label':name,'x':[r['epoch'] for r in plain[source]],'y':[r[field]*100 for r in plain[source]]}
            for name,source,field in [('训练','history','train_error'),('验证','history','val_error'),('测试','tests','test_error')]]
    replacements['OVERFIT_PLOTS']=pair(
        plot(errors,'B0：误差与泛化差距','分类误差（\%）',
             options='xmin=1,xmax=60,ymin=0,ymax=18,xtick={1,10,20,30,40,50,60}',legend_columns=3),
        plot(lines(regularization,'val_loss'),'验证数据损失（CE）','交叉熵',
             options='xmin=1,xmax=60,ymin=0,ymax=.85,xtick={1,10,20,30,40,50,60}',legend_columns=3))
    replacements['VALLOSS_ROWS']=''.join(table_row(NAMES[key],f"{RUNS[key]['history'][-1]['val_loss']:.4f}",pct(best(key)[0]['val_error'])) for key in regularization)
    replacements['CONFUSION_PLOTS']=confusion()
    replacements['CONFUSION_ROWS']=''.join(table_row(E['classes'][item['true']]['chinese'],E['classes'][item['pred']]['chinese'],item['count']) for item in E['top_confusions'][:5])
    replacements['ERROR_SAMPLES']=image_grid(E['error_examples'],True)
    replacements['XOR_ROWS']=''.join(table_row(x1,x2,target,f'{value:.9f}',prediction)
                                    for (x1,x2,target),value,prediction in zip([(0,0,0),(0,1,1),(1,0,1),(1,1,0)],E['xor']['predictions'],E['xor']['classes']))
    replacements['XOR_PLOTS']=xor_plot()

    appendix=[]
    for i,key in enumerate(KEYS):
        run=RUNS[key]
        history=run['history']
        summary=run['summary']
        loss_type='MSE' if key=='mse' else 'CE'
        series=lines([key],'train_loss',labels=['训练数据'])+lines([key],'val_loss',labels=['验证数据'])
        if summary['config']['l2']>0:
            series+=lines([key],'train_total_loss',labels=['训练总目标'])
        opts=f"xmin=1,xmax={summary['completed_epochs']},ymin=0"
        left=plot(series,f'损失（{loss_type}）','Loss',options=opts,height='4.25cm',legend_columns=3)
        errors=[{'label':name,'x':[r['epoch'] for r in run[source]],'y':[r[field]*100 for r in run[source]]}
                for name,source,field in [('训练','history','train_error'),('验证','history','val_error'),('测试','tests','test_error')]]
        right=plot(errors,'分类误差','误差（\%）',options=opts+',ymax=18',height='4.25cm',legend_columns=3,best_epoch=summary['best_epoch'])
        appendix.append(r'\begin{figure}[H]\centering'+pair(left,right)
                        +r'\caption{'+IDS[key]+'：'+NAMES[key]+f"。共 {summary['completed_epochs']} 轮，参数量 {summary['num_params']:,}，验证选中第 {summary['best_epoch']} 轮。"
                        +r'}\end{figure}'+'\n')
        if (i+1)%3==0 and i+1<len(KEYS):
            appendix.append(r'\clearpage'+'\n')
    replacements['ALL_CURVES']=''.join(appendix)
    replacements.update(question_content())
    text=(ROOT/'report_template.tex').read_text(encoding='utf-8')
    for key,value in replacements.items():
        text=text.replace('@@'+key+'@@',value)
    unresolved=re.findall(r'@@[A-Z_]+@@',text)
    if unresolved:
        raise ValueError(f'Unresolved placeholders: {unresolved}')
    target=ROOT/'2412526-史峰源-实验1.tex'
    target.write_text(text,encoding='utf-8')
    print(f'Created {target.name}; {len(text):,} characters; {text.count(chr(92)+"caption{")} figures/tables.')


if __name__=='__main__':
    main()
