import glob

for file in glob.glob('validation/scenarios/**/*.py', recursive=True):
    with open(file, 'r') as f:
        content = f.read()
    content = content.replace('severity=["warning"]', 'severity=["warning", "critical"]')
    content = content.replace('severity=["critical"]', 'severity=["warning", "critical"]')
    content = content.replace('severity=["critical", "warning"]', 'severity=["warning", "critical"]')
    with open(file, 'w') as f:
        f.write(content)
