import os
import re

bin_dir = r"c:\Users\Juwan\Desktop\RAVEN\src\bin"

for fname in os.listdir(bin_dir):
    if not fname.endswith(".rs"):
        continue
    fpath = os.path.join(bin_dir, fname)
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
    
    if "VamanaBuildConfig" not in content:
        continue
    if "enable_layered_nav" in content:
        continue  # already fixed
    
    # Pattern: after max_iterations: <value>, add ..Default::default(),
    # but only if it's inside a VamanaBuildConfig block
    # We look for the pattern: max_iterations: <val>,\n        };
    # and replace with: max_iterations: <val>,\n        ..Default::default(),\n        };
    
    # More robust: find all VamanaBuildConfig { ... } blocks
    # and add ..Default::default() before the closing };
    
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
