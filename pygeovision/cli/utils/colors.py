"""Terminal color helpers."""
GREEN  = "\033[92m"; RED    = "\033[91m"; YELLOW = "\033[93m"
BLUE   = "\033[94m"; CYAN   = "\033[96m"; BOLD   = "\033[1m"
RESET  = "\033[0m";  DIM    = "\033[2m"

def ok(msg):  return f"{GREEN}✓{RESET} {msg}"
def err(msg): return f"{RED}✗{RESET} {msg}"
def warn(msg):return f"{YELLOW}⚠{RESET} {msg}"
def info(msg):return f"{BLUE}ℹ{RESET} {msg}"
