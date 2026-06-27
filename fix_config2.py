import os
import re

bin_dir = r"c:\Users\Juwan\Desktop\RAVEN\src\bin"

for fname in sorted(os.listdir(bin_dir)):
    if not fname.endswith(".rs"):
        continue
    fpath = os.path.join(bin_dir, fname)
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        # Try with latin-1 and re-encode
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    
    if "VamanaBuildConfig" not in content:
        continue
    if "enable_layered_nav" in content:
        continue  # already fixed
    
    pattern = r'(max_iterations:\s*[^,\n]+,)\s*\n(\s*\};)'
    
    def replacer(m):
        return m.group(1) + "\n..Default::default(),\n" + m.group(2)
    
    new_content = re.sub(pattern, replacer, content)
    
    if new_content != content:
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Fixed: {fname}")
    else:
        print(f"No match: {fname}")
