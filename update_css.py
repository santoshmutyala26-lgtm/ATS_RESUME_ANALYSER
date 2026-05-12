import re
with open('static/css/style.css', 'r') as f:
    css = f.read()
root_pattern = ':root\\s*\\{[^}]*\\}'
new_root = ':root {\n  --bg:\n  --bg2:\n  --bg3:\n  --bg4:\n  --border:   rgba(0,0,0,0.08);\n  --border2:  rgba(0,0,0,0.15);\n  --purple:\n  --purple2:\n  --green:\n  --amber:\n  --red:\n  --blue:\n  --text1:\n  --text2:\n  --text3:\n  --radius:   8px;\n  --radius-sm:6px;\n  --radius-lg:12px;\n}'
css = re.sub(root_pattern, new_root, css)
css = css.replace('rgba(255,255,255,0.06)', 'rgba(0,0,0,0.04)')
css = css.replace('rgba(255,255,255,0.12)', 'rgba(0,0,0,0.1)')
css = css.replace('rgba(13,13,20,0.85)', 'rgba(255,255,255,0.95)')
css = css.replace('rgba(108,99,255,0.15)', 'rgba(0,0,0,0.05)')
css = css.replace('rgba(108,99,255,0.3)', 'rgba(0,0,0,0.15)')
css = css.replace('rgba(108,99,255,0.06)', 'rgba(0,0,0,0.03)')
css = css.replace('rgba(108,99,255,0.35)', 'rgba(0,0,0,0.2)')
css = css.replace('rgba(108,99,255,0.25)', 'rgba(0,0,0,0.1)')
css = css.replace('rgba(108,99,255,0.1)', 'rgba(0,0,0,0.06)')
css = css.replace('rgba(108,99,255,0.12)', 'rgba(0,0,0,0.04)')
css = css.replace('color: #4ade80;', 'color: #15803d;')
css = css.replace('color: #f87171;', 'color: #b91c1c;')
css = css.replace('color: #fbbf24;', 'color: #b45309;')
css = css.replace('color: #fca5a5;', 'color: #b91c1c;')
css = css.replace('color: #fde68a;', 'color: #b45309;')
css = css.replace('color: #86efac;', 'color: #15803d;')
css = css.replace('stroke="#1e1e2e"', 'stroke="#e5e7eb"')
css = css.replace('background: var(--bg2);\n  border: 1px solid var(--border2);', 'background: var(--bg2);\n  border: 1px solid var(--border2);\n  box-shadow: 0 4px 20px rgba(0,0,0,0.03);')
css = css.replace('background: var(--purple); color: white;', 'background: var(--purple); color: white;')
with open('static/css/style.css', 'w') as f:
    f.write(css)
print('CSS updated to minimalist theme!')
