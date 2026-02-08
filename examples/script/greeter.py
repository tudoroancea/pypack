"""A simple single-file script example."""

import sys
import json

data = {"greeting": "Hello", "target": sys.argv[1] if len(sys.argv) > 1 else "world"}
print(json.dumps(data, indent=2))
