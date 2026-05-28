from flask import Flask, jsonify, request, render_template
import pandas as pd
import numpy as np
import io
import sys
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

app = Flask(__name__)

MODELOS_POOL = {}
SCALER_GLOBAL = None
FEATURES_ML = []
CLASES_TARGET = []

def inicializar_entorno_ia():
    global MODELOS_POOL, SCALER_GLOBAL, FEATURES_ML, CLASES_TARGET
    wine = load_wine()
    X, y = wine.data, wine.target
    FEATURES_ML = list(wine.feature_names)
    CLASES_TARGET = list(wine.target_names)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    SCALER_GLOBAL = StandardScaler()
    X_train_scaled = SCALER_GLOBAL.fit_transform(X_train)
    X_test_scaled = SCALER_GLOBAL.transform(X_test)
    X_all_scaled = SCALER_GLOBAL.transform(X)

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

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/ml-results")
def get_ml_results():
    return jsonify(CACHED_DATA)

@app.route("/api/predict-table", methods=["POST"])
def predict_table():
    try:
        data = request.get_json()
        raw_df = pd.DataFrame(data.get('rows', []))
        
        # 1. ELIMINACIÓN DE DUPLICADOS: barremos cualquier columna predictiva previa enviada por Vue
        columnas_limpias = [c for c in raw_df.columns if not c.startswith('PRED_')]
        df_base = raw_df[columnas_limpias].copy()
        
        # 2. RESOLUCIÓN DE SETTINGWITHCOPY: Usamos .copy() explícito para aislar df_infer de la memoria
        df_infer = df_base.copy()
        
        for f in FEATURES_ML:
            if f not in df_infer.columns:
                df_infer[f] = 0.0
                
        # Forzamos conversión numérica estricta alineada a las 13 características
        df_infer = df_infer[FEATURES_ML].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        
        X_scaled = SCALER_GLOBAL.transform(df_infer.to_numpy())
        
        # 3. CONCATENACIÓN CON LLAVES ÚNICAS: Generamos la matriz predictiva limpia
        df_preds = pd.DataFrame(index=df_base.index)
        for name, model in MODELOS_POOL.items():
            preds = model.predict(X_scaled)
            col_name = f"PRED_{name.upper().replace(' ', '_')}"
            df_preds[col_name] = [CLASES_TARGET[p] for p in preds]
            
        # Unimos predicciones únicas y variables de entrada de forma lineal
        df_final = pd.concat([df_preds, df_base], axis=1)
        
        return jsonify({
            "status": "ok",
            "headers": list(df_final.columns),
            "rows": df_final.replace({np.nan: None}).to_dict(orient='records')
        })
    except Exception as e:
        print(f"💥 [CRITICAL] Error en predict-table: {str(e)}", file=sys.stderr)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/upload-csv", methods=["POST"])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "Falta el archivo"}), 400
    file = request.files['file']
    try:
        contenido = file.stream.read().decode("utf-8", errors='replace')
        df = pd.read_csv(io.StringIO(contenido, newline=None), sep=None, engine='python')
        
        # Limpieza preventiva de cabeceras previas en archivos planos
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
            col_name = f"PRED_{name.upper().replace(' ', '_')}"
            df_preds[col_name] = [CLASES_TARGET[p] for p in preds]

        df_final = pd.concat([df_preds, df_base], axis=1)
        df_display = df_final.head(100).replace({np.nan: None})
        
        return jsonify({
            "status": "ok",
            "headers": list(df_display.columns),
            "rows": df_display.to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)