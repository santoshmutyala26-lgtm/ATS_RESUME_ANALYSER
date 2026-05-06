import re

with open('static/css/style.css', 'r') as f:
    css = f.read()

# 1. Replace :root variables with Minimalist Light Theme
root_pattern = r':root\s*\{[^}]*\}'
new_root = """:root {
  --bg:       #ffffff;
  --bg2:      #ffffff;
  --bg3:      #f9fafb;
  --bg4:      #f3f4f6;
  --border:   rgba(0,0,0,0.08);
  --border2:  rgba(0,0,0,0.15);
  --purple:   #000000; /* Black for buttons/accents */
  --purple2:  #4b5563; /* Dark gray */
  --green:    #10b981;
  --amber:    #f59e0b;
  --red:      #ef4444;
  --blue:     #3b82f6;
  --text1:    #111827;
  --text2:    #4b5563;
  --text3:    #6b7280;
  --radius:   8px;
  --radius-sm:6px;
  --radius-lg:12px;
}"""
css = re.sub(root_pattern, new_root, css)

# 2. Fix hardcoded translucent whites -> translucent blacks
css = css.replace('rgba(255,255,255,0.06)', 'rgba(0,0,0,0.04)')
css = css.replace('rgba(255,255,255,0.12)', 'rgba(0,0,0,0.1)')

# 3. Fix Nav Background
css = css.replace('rgba(13,13,20,0.85)', 'rgba(255,255,255,0.95)')

# 4. Fix hardcoded purple rgba -> black/gray rgba
css = css.replace('rgba(108,99,255,0.15)', 'rgba(0,0,0,0.05)')
css = css.replace('rgba(108,99,255,0.3)', 'rgba(0,0,0,0.15)')
css = css.replace('rgba(108,99,255,0.06)', 'rgba(0,0,0,0.03)')
css = css.replace('rgba(108,99,255,0.35)', 'rgba(0,0,0,0.2)')
css = css.replace('rgba(108,99,255,0.25)', 'rgba(0,0,0,0.1)')
css = css.replace('rgba(108,99,255,0.1)', 'rgba(0,0,0,0.06)')
css = css.replace('rgba(108,99,255,0.12)', 'rgba(0,0,0,0.04)')

# 5. Fix hardcoded text colors in specific tags that expect dark mode
css = css.replace('color: #4ade80;', 'color: #15803d;') # light green -> dark green
css = css.replace('color: #f87171;', 'color: #b91c1c;') # light red -> dark red
css = css.replace('color: #fbbf24;', 'color: #b45309;') # light amber -> dark amber
css = css.replace('color: #fca5a5;', 'color: #b91c1c;')
css = css.replace('color: #fde68a;', 'color: #b45309;')
css = css.replace('color: #86efac;', 'color: #15803d;')

# 6. Loading ring background
css = css.replace('stroke="#1e1e2e"', 'stroke="#e5e7eb"')

# 7. Specific element tweaks for Light Mode contrast
# The "Upload Resume" and "Paste Job Desc" inputs should have stronger borders
css = css.replace('background: var(--bg2);\n  border: 1px solid var(--border2);', 'background: var(--bg2);\n  border: 1px solid var(--border2);\n  box-shadow: 0 4px 20px rgba(0,0,0,0.03);')
css = css.replace('background: var(--purple); color: white;', 'background: var(--purple); color: white;') # Actually fine as black and white

with open('static/css/style.css', 'w') as f:
    f.write(css)

print("CSS updated to minimalist theme!")
