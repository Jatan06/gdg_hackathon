import os
if os.path.exists("dispatch.db"):
    os.remove("dispatch.db")
    print("Old database deleted successfully.")
else:
    print("Database not found.")
