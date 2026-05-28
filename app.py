from flask import Flask, jsonify, request, render_template
import pandas as pd
import numpy as np
import io
import sys
import time
import optuna
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

optuna.logging.set_verbosity(optuna.logging.WARNING)

app = Flask(__name__)

MODELOS_POOL = {}
SCALER_GLOBAL = None
FEATURES_ML = []
CLASES_TARGET = []
X_TRAIN_SCALED, X_TEST_SCALED, Y_TRAIN, Y_TEST = None, None, None, None

def inicializar_entorno_ia():
    global MODELOS_POOL, SCALER_GLOBAL, FEATURES_ML, CLASES_TARGET
    global X_TRAIN_SCALED, X_TEST_SCALED, Y_TRAIN, Y_TEST
    
    wine = load_wine()
    X, y = wine.data, wine.target
    FEATURES_ML = list(wine.feature_names)
    CLASES_TARGET = list(wine.target_names)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    Y_TRAIN, Y_TEST = y_train, y_test

    SCALER_GLOBAL = StandardScaler()
    X_train_scaled = SCALER_GLOBAL.fit_transform(X_train)
    X_test_scaled = SCALER_GLOBAL.transform(X_test)
    X_all_scaled = SCALER_GLOBAL.transform(X)
    
    X_TRAIN_SCALED, X_TEST_SCALED = X_train_scaled, X_test_scaled

    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X_train_scaled, y_train)
    xgb = XGBClassifier(n_estimators=50, max_depth=3, objective='multi:softprob', random_state=42).fit(X_train_scaled, y_train)
    nb = GaussianNB().fit(X_train_scaled, y_train)
    softmax = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200, random_state=42).fit(X_train_scaled, y_train)

    MODELOS_POOL = {"Random Forest": rf, "XGBoost": xgb, "Naive Bayes": nb, "Softmax Regression": softmax}

    modelos_stats = {}
    for name, model in MODELOS_POOL.items():
        modelos_stats[name] = {
            "train_acc": f"{accuracy_score(y_train, model.predict(X_train_scaled))*100:.2f}%",
            "test_acc": f"{accuracy_score(y_test, model.predict(X_test_scaled))*100:.2f}%",
            "precision": f"{precision_score(y_test, model.predict(X_test_scaled), average='macro')*100:.2f}%",
            "recall": f"{recall_score(y_test, model.predict(X_test_scaled), average='macro')*100:.2f}%"
        }

    pca_coor = PCA(n_components=2).fit_transform(X_all_scaled)
    tsne_coor = TSNE(n_components=2, perplexity=15, random_state=42).fit_transform(X_all_scaled)
    umap_coor = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(X_all_scaled)

    puntos = []
    for i in range(len(X)):
        puntos.append({
            "clase": CLASES_TARGET[y[i]],
            "pca_x": float(pca_coor[i, 0]), "pca_y": float(pca_coor[i, 1]),
            "tsne_x": float(tsne_coor[i, 0]), "tsne_y": float(tsne_coor[i, 1]),
            "umap_x": float(umap_coor[i, 0]), "umap_y": float(umap_coor[i, 1])
        })

    return {"modelos": modelos_stats, "clases": CLASES_TARGET, "puntos": puntos}

CACHED_DATA = inicializar_entorno_ia()

def evaluar_tipo_estricto(v_str):
    v = v_str.strip()
    if v.lower() == 'true': return True
    if v.lower() == 'false': return False
    try:
        if 'e' in v.lower() or '.' in v: return float(v)
        return int(v)
    except ValueError:
        return v

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/ml-results")
def get_ml_results():
    return jsonify(CACHED_DATA)

@app.route("/api/retrain", methods=["POST"])
def retrain_and_benchmark():
    try:
        req = request.get_json()
        modelo_seleccionado = req.get("model")
        grid_raw_text = req.get("grid_text", "")

        param_grid = {}
        for linea in grid_raw_text.split("\n"):
            if ":" in linea:
                clave, valores = linea.split(":", 1)
                param_grid[clave.strip()] = [evaluar_tipo_estricto(x) for x in valores.split(",")]

        if modelo_seleccionado == "Random Forest":
            estimator = RandomForestClassifier(random_state=42)
        elif modelo_seleccionado == "XGBoost":
            estimator = XGBClassifier(objective='multi:softprob', random_state=42)
        elif modelo_seleccionado == "Softmax Regression":
            estimator = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500, random_state=42)
        else:
            estimator = GaussianNB()

        resultados_lista = []

        # 1. GRID SEARCH
        t_start = time.perf_counter()
        gs = GridSearchCV(estimator, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        gs.fit(X_TRAIN_SCALED, Y_TRAIN)
        t_grid = time.perf_counter() - t_start
        resultados_lista.append({
            "estrategia": "Grid Search (Fuerza Bruta)",
            "tiempo": f"{t_grid:.4f}s",
            "test_acc": f"{accuracy_score(Y_TEST, gs.best_estimator_.predict(X_TEST_SCALED))*100:.2f}%"
        })

        # 2. RANDOM SEARCH
        t_start = time.perf_counter()
        try:
            max_comb = int(np.prod([len(v) for v in param_grid.values()]))
        except Exception:
            max_comb = 1
        n_iteraciones = max(1, min(8, max_comb))
        
        rs = RandomizedSearchCV(estimator, param_grid, n_iter=n_iteraciones, cv=3, scoring='accuracy', n_jobs=-1, random_state=42)
        rs.fit(X_TRAIN_SCALED, Y_TRAIN)
        t_random = time.perf_counter() - t_start
        resultados_lista.append({
            "estrategia": "Random Search (Muestreo)",
            "tiempo": f"{t_random:.4f}s",
            "test_acc": f"{accuracy_score(Y_TEST, rs.best_estimator_.predict(X_TEST_SCALED))*100:.2f}%"
        })

        # 3. OPTUNA
        t_start = time.perf_counter()
        def objective(trial):
            params = {}
            for param_name, valores in param_grid.items():
                if any(isinstance(x, str) or isinstance(x, bool) for x in valores):
                    params[param_name] = trial.suggest_categorical(param_name, valores)
                else:
                    if len(valores) == 1:
                        params[param_name] = valores[0]
                    else:
                        p_min, p_max = min(valores), max(valores)
                        if p_min == p_max:
                            params[param_name] = p_min
                        elif any(isinstance(x, float) for x in valores) or 'e' in str(p_min).lower():
                            params[param_name] = trial.suggest_float(param_name, float(p_min), float(p_max))
                        else:
                            params[param_name] = trial.suggest_int(param_name, int(p_min), int(p_max))

            if modelo_seleccionado == "Random Forest":
                clf = RandomForestClassifier(**params, random_state=42)
            elif modelo_seleccionado == "XGBoost":
                clf = XGBClassifier(**params, objective='multi:softprob', random_state=42)
            elif modelo_seleccionado == "Softmax Regression":
                clf = LogisticRegression(**params, multi_class='multinomial', solver='lbfgs', max_iter=500, random_state=42)
            else:
                clf = GaussianNB(**params)
            return cross_val_score(clf, X_TRAIN_SCALED, Y_TRAIN, cv=3, scoring='accuracy').mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10)
        t_optuna = time.perf_counter() - t_start

        if modelo_seleccionado == "Random Forest":
            opt_clf = RandomForestClassifier(**study.best_params, random_state=42).fit(X_TRAIN_SCALED, Y_TRAIN)
        elif modelo_seleccionado == "XGBoost":
            opt_clf = XGBClassifier(**study.best_params, objective='multi:softprob', random_state=42).fit(X_TRAIN_SCALED, Y_TRAIN)
        elif modelo_seleccionado == "Softmax Regression":
            opt_clf = LogisticRegression(**study.best_params, multi_class='multinomial', solver='lbfgs', max_iter=500, random_state=42).fit(X_TRAIN_SCALED, Y_TRAIN)
        else:
            opt_clf = GaussianNB(**study.best_params).fit(X_TRAIN_SCALED, Y_TRAIN)

        resultados_lista.append({
            "estrategia": "Optuna (Muestreo Bayesiano)",
            "tiempo": f"{t_optuna:.4f}s",
            "test_acc": f"{accuracy_score(Y_TEST, opt_clf.predict(X_TEST_SCALED))*100:.2f}%"
        })

        # CLONACIÓN DEL MODELO: En lugar de reemplazar el viejo, añadimos el optimizado al diccionario
        nombre_optimizado = f"{modelo_seleccionado} (Optimizado)"
        MODELOS_POOL[nombre_optimizado] = gs.best_estimator_
        
        y_pred = gs.best_estimator_.predict(X_TEST_SCALED)
        y_train_pred = gs.best_estimator_.predict(X_TRAIN_SCALED)
        
        # Guardamos estadísticas bajo la nueva llave del pool sin pisar el modelo base original
        CACHED_DATA["modelos"][nombre_optimizado] = {
            "train_acc": f"{accuracy_score(Y_TRAIN, y_train_pred)*100:.2f}%",
            "test_acc": resultados_lista[0]["test_acc"],
            "precision": f"{precision_score(Y_TEST, y_pred, average='macro')*100:.2f}%",
            "recall": f"{recall_score(Y_TEST, y_pred, average='macro')*100:.2f}%",
            "es_optimizado": True  # Flag para pintar las letras verdes en el frontend
        }

        pool_convertido_lista = []
        for name, stats in CACHED_DATA["modelos"].items():
            pool_convertido_lista.append({
                "arquitectura": name,
                "train_acc": stats["train_acc"],
                "test_acc": stats["test_acc"],
                "precision": stats["precision"],
                "es_optimizado": stats.get("es_optimizado", False)
            })

        return jsonify({"status": "ok", "benchmark_lista": resultados_lista, "pool_lista": pool_convertido_lista})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/predict-table", methods=["POST"])
def predict_table():
    try:
        data = request.get_json()
        raw_df = pd.DataFrame(data.get('rows', []))
        columnas_limpias = [c for c in raw_df.columns if not c.startswith('PRED_')]
        df_base = raw_df[columnas_limpias].copy()
        df_infer = df_base.copy()
        for f in FEATURES_ML:
            if f not in df_infer.columns:
                df_infer[f] = 0.0
        df_infer = df_infer[FEATURES_ML].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        X_scaled = SCALER_GLOBAL.transform(df_infer.to_numpy())
        
        df_preds = pd.DataFrame(index=df_base.index)
        for name, model in MODELOS_POOL.items():
            preds = model.predict(X_scaled)
            col_name = f"PRED_{name.upper().replace(' ', '_').replace('_(OPTIMIZADO)', '_OPT')}"
            df_preds[col_name] = [CLASES_TARGET[p] for p in preds]
            
        df_final = pd.concat([df_preds, df_base], axis=1)
        return jsonify({"status": "ok", "headers": list(df_final.columns), "rows": df_final.replace({np.nan: None}).to_dict(orient='records')})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/upload-csv", methods=["POST"])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "Falta el archivo"}), 400
    file = request.files['file']
    try:
        contenido = file.stream.read().decode("utf-8", errors='replace')
        df = pd.read_csv(io.StringIO(contenido, newline=None), sep=None, engine='python')
        columnas_csv = [c.strip() for c in df.columns if not c.strip().startswith('PRED_')]
        df_base = df[columnas_csv].copy()
        
        if all(elem in columnas_csv for elem in FEATURES_ML):
            df_infer = df_base[FEATURES_ML].copy()
            matriz_numpy = df_infer.to_numpy()
        else:
            registros_archivo = len(df_base) if len(df_base) > 0 else 1
            matriz_numpy = np.tile(SCALER_GLOBAL.mean_, (registros_archivo, 1))
            for idx, feat in enumerate(FEATURES_ML):
                if feat not in df_base.columns:
                    df_base[feat] = matriz_numpy[:, idx]

        X_scaled = SCALER_GLOBAL.transform(matriz_numpy.astype(float))
        df_preds = pd.DataFrame(index=df_base.index)
        for name, model in MODELOS_POOL.items():
            preds = model.predict(X_scaled)
            col_name = f"PRED_{name.upper().replace(' ', '_').replace('_(OPTIMIZADO)', '_OPT')}"
            df_preds[col_name] = [CLASES_TARGET[p] for p in preds]

        df_final = pd.concat([df_preds, df_base], axis=1)
        df_display = df_final.head(100).replace({np.nan: None})
        return jsonify({"status": "ok", "headers": list(df_display.columns), "rows": df_display.to_dict(orient='records')})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)