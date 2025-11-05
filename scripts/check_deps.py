import importlib
packages = ['flask','flask_cors','pandas','numpy','joblib','sklearn','matplotlib','seaborn','tensorflow','shap','lime']
for p in packages:
    try:
        importlib.import_module(p)
        print(f"{p}: OK")
    except Exception as e:
        print(f"{p}: MISSING - {e.__class__.__name__}: {e}")
