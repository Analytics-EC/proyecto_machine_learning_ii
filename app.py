import uvicorn
import logging
from typing import Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import joblib
import xgboost as xgb
import pandas as pd
import numpy as np
import shap

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.metrics import accuracy_score, precision_score, recall_score

from enum import Enum
from pydantic import BaseModel, Field


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ==========================================
# CONFIGURACIÓN DE RUTAS ABSOLUTAS (BLINDADO)
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
CATEGORIAS_ORDENADAS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
TIPO_ORDINAL_XGB = pd.api.types.CategoricalDtype(categories=CATEGORIAS_ORDENADAS, ordered=True)

app = FastAPI(
    title='NASA Asteroids Machine Learning & XAI API',
    description='Backend refactored to FastAPI without scaling, matching original notebook conditions.',
    version='2.1.0'
)

# Validar físicamente la existencia antes de montar
if not STATIC_DIR.exists():
    raise RuntimeError(f"❌ ERROR CRÍTICO: La carpeta física 'static' NO EXISTE en: {STATIC_DIR}")
if not TEMPLATES_DIR.exists():
    raise RuntimeError(f"❌ ERROR CRÍTICO: La carpeta física 'templates' NO EXISTE en: {TEMPLATES_DIR}")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


class ModelNameEnum(str, Enum):
    RANDOM_FOREST = "Random Forest"
    XGBOOST = "XGBoost"

class AsteroidPredictRequest(BaseModel):
    model_name: ModelNameEnum = Field(..., description="Nombre del modelo de ML a ejecutar.")
    absolute_magnitude: float = Field(..., description="Magnitud absoluta H del objeto.")
    relative_velocity_km_per_hr: float = Field(..., description="Velocidad relativa en km/h.")
    miss_dist_kilometers: float = Field(..., description="Distancia mínima de aproximación en km.")
    orbit_uncertainity: int = Field(..., ge=0, le=9, description="Código de incertidumbre de la órbita (0-9).")
    minimum_orbit_intersection: float = Field(..., description="Distancia mínima de intersección orbital (MOID).")
    eccentricity: float = Field(..., description="Excentricidad de la órbita.")
    semi_major_axis: float = Field(..., description="Semieje mayor de la órbita.")
    inclination: float = Field(..., description="Inclinación orbital en grados.")
    asc_node_longitude: float = Field(..., description="Longitud del nodo ascendente.")
    perihelion_distance: float = Field(..., description="Distancia al perihelio.")
    perihelion_arg: float = Field(..., description="Argumento del perihelio.")
    mean_anomaly: float = Field(..., description="Anomalía media.")
    is_coplanar: bool = Field(..., description="Indicador de órbita coplanar.")


# ==========================================
# POOL DE DATOS Y VARIABLES GLOBALES (SIN SCALER)
# ==========================================
MODELOS_POOL = {}
CLASES_TARGET = ["No Peligroso", "Peligroso"]
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = None, None, None, None
CACHED_DATA = {}
FEATURES_ML = []

def inicializar_entorno_ia():
    global MODELOS_POOL, FEATURES_ML, CLASES_TARGET
    global X_TRAIN, X_TEST, Y_TRAIN, Y_TEST
    
    FEATURES_ML = [
        'absolute_magnitude', 'relative_velocity_km_per_hr', 'miss_dist_kilometers',
        'orbit_uncertainity', 'minimum_orbit_intersection', 'eccentricity',
        'semi_major_axis', 'inclination', 'asc_node_longitude',
        'perihelion_distance', 'perihelion_arg', 'mean_anomaly', 'is_coplanar'
    ]
    path_feather = STATIC_DIR / 'data_feature_engineering.feather'
    path_rf = STATIC_DIR / 'random_forest.joblib'
    path_xgb = STATIC_DIR / 'xgboost.json'
    target_col = "hazardous"
    
    # Carga del Dataset
    df = pd.read_feather(str(path_feather))
    logger.info(f"df types: {df.dtypes}")
    if "neo_reference_id" in df.columns:
        df = df.drop(columns=["neo_reference_id"])
    if "close_approach_date" in df.columns:
        df = df.drop(columns=["close_approach_date"])
        
    X = df[FEATURES_ML]
    y = df[target_col]
    
    # Partición exacta (Sin transformaciones ni ajustes de escala redundantes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    X_TRAIN = X_train
    X_TEST = X_test
    Y_TRAIN = y_train
    Y_TEST = y_test
    
    # Carga de artefactos binarios originales
    rf_model = joblib.load(str(path_rf))
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(path_xgb))
    
    MODELOS_POOL = {
        "Random Forest": rf_model,
        "XGBoost": xgb_model
    }
    
    modelos_stats = {}
    for name, model in MODELOS_POOL.items():
        try:
            preds_train = model.predict(X_TRAIN)
            preds_test = model.predict(X_TEST)
            modelos_stats[name] = {
                'train_acc': f'{accuracy_score(Y_TRAIN, preds_train)*100:.2f}%',
                'test_acc': f'{accuracy_score(Y_TEST, preds_test)*100:.2f}%',
                'precision': f'{precision_score(Y_TEST, preds_test, average="macro", zero_division=0)*100:.2f}%',
                'recall': f'{recall_score(Y_TEST, preds_test, average="macro", zero_division=0)*100:.2f}%'
            }
        except Exception as e:
            modelos_stats[name] = {'error': str(e)}
            
    return {'modelos': modelos_stats, 'clases': CLASES_TARGET}

# Ejecutar Setup inicial
CACHED_DATA = inicializar_entorno_ia()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get('/api/ml-results')
async def get_ml_results():
    return CACHED_DATA

import pandas.api.types as pjs

@app.get('/api/dataset-stats')
async def get_dataset_stats():
    """Analiza las columnas del dataset, detectando tipos continuos, discretos y booleanos"""
    try:
        path_feather = STATIC_DIR / "data_feature_engineering.feather"
        df = pd.read_feather(str(path_feather))
        stats = {}
        
        for feat in FEATURES_ML:
            if feat in df.columns:
                # 1. Detectar si el campo es de naturaleza booleana / flag lógico
                is_bool = pjs.is_bool_dtype(df[feat]) or feat in ["is_coplanar"]
                
                # 2. Detectar si es un entero o categoría discreta (como orbit_uncertainity)
                is_discrete = (
                    df[feat].dtype.name == 'category' or 
                    pjs.is_integer_dtype(df[feat]) or 
                    feat == "orbit_uncertainity"
                )
                
                stats[feat] = {
                    "min": 0 if is_bool else (int(df[feat].min()) if is_discrete else float(df[feat].min())),
                    "max": 1 if is_bool else (int(df[feat].max()) if is_discrete else float(df[feat].max())),
                    "default": bool(df[feat].iloc[0]) if is_bool else (int(df[feat].iloc[0]) if is_discrete else float(df[feat].iloc[0])),
                    "is_bool": is_bool,
                    "step": 1 if is_discrete else "any"
                }
        return {"status": "ok", "features": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/dataset-table')
async def get_dataset_table(page: int = 1, per_page: int = 10):
    try:
        path_feather = STATIC_DIR / "data_feature_engineering.feather"
        df = pd.read_feather(str(path_feather))
        total_rows = len(df)
        start = (page - 1) * per_page
        end = start + per_page
        
        cols_to_show = FEATURES_ML + ["hazardous"]
        sub_df = df[cols_to_show].iloc[start:end].copy()
        
        if "is_coplanar" in sub_df.columns:
            sub_df["is_coplanar"] = sub_df["is_coplanar"].astype(str)
            
        data_rows = sub_df.to_dict(orient="records")
        return {
            "status": "ok",
            "total": total_rows,
            "page": page,
            "per_page": per_page,
            "rows": data_rows
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/api/xai/permutation')
async def get_permutation_importance(payload: dict[str, Any]):
    try:
        name = payload.get('model', 'Random Forest')
        base_name = name.replace(' (Optimizado)', '')
        model = MODELOS_POOL.get(name) or MODELOS_POOL.get(base_name)

        if not model:
            raise HTTPException(status_code=400, detail='Modelo no encontrado')
            
        # Ejecución sobre los datos crudos originales
        r = permutation_importance(model, X_TEST, Y_TEST, n_repeats=5, random_state=42, n_jobs=-1)
        lista_importancias = []
        for idx, feat in enumerate(FEATURES_ML):
            lista_importancias.append({
                'feature': feat,
                'importance_mean': float(r.importances_mean[idx]),
                'importance_std': float(r.importances_std[idx])
            })
        lista_importancias = sorted(lista_importancias, key=lambda x: x['importance_mean'], reverse=True)
        return {'status': 'ok', 'permutations': lista_importancias}
    except HTTPException as he: raise he
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.post('/api/xai/pdp-ice')
async def get_pdp_ice_data(payload: dict[str, Any]):
    try:
        model_name = payload.get('model', 'Random Forest')
        feature_name = payload.get('feature')
        if not feature_name and FEATURES_ML: feature_name = FEATURES_ML[0]
        base_name = model_name.replace(' (Optimizado)', '')
        model = MODELOS_POOL.get(model_name) or MODELOS_POOL.get(base_name)
        
        if not model or feature_name not in FEATURES_ML:
            raise HTTPException(status_code=400, detail='Parámetros inválidos')
            
        feat_idx = FEATURES_ML.index(feature_name)
        # PDP ejecutado sobre X_TRAIN original para respetar los límites físicos reales de los sliders
        pdp_res = partial_dependence(model, X_TRAIN, features=[feat_idx], kind='both', grid_resolution=25)
        return {
            'status': 'ok',
            'grid': pdp_res['grid_values'][0].tolist(),
            'pdp': pdp_res['average'][0].tolist(),
            'ice': pdp_res['individual'][0].tolist()[:30]
        }
    except HTTPException as he: raise he
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.post('/api/xai/shap')
async def get_shap_values(payload: dict[str, Any]):

    try:
        model_name = payload.get('model', 'Random Forest')
        base_name = model_name.replace(' (Optimizado)', '')
        model = MODELOS_POOL.get(model_name) or MODELOS_POOL.get(base_name)
        if not model: raise HTTPException(status_code=400, detail='Modelo inválido')
        
        # Muestra tomada directamente de los datos reales para el gráfico sumario de SHAP
        X_sample = X_TEST[:60]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        if isinstance(shap_values, list):
            vals = np.array(shap_values[1] if len(shap_values) > 1 else shap_values[0], dtype=np.float64)
        elif hasattr(shap_values, 'values'):
            v_raw = shap_values.values
            vals = np.array(v_raw[:, :, 1] if len(v_raw.shape) == 3 else v_raw, dtype=np.float64)
        else:
            shap_arr = np.array(shap_values, dtype=np.float64)
            vals = shap_arr[:, :, 1] if len(shap_arr.shape) == 3 else shap_arr
            
        return {
            'status': 'ok',
            'features': FEATURES_ML,
            'shap_values': vals.tolist(),
            'real_values': X_sample.values.tolist()
        }
    except HTTPException as he: raise he
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict")
async def predict(payload: AsteroidPredictRequest):
    try:
        selected_model_name = payload.model_name.value
        model = MODELOS_POOL.get(selected_model_name)
        
        if not model:
            raise HTTPException(status_code=400, detail=f"El modelo '{selected_model_name}' no está cargado.")
            
        data_dict = payload.model_dump()
        input_row = {feat: data_dict[feat] for feat in FEATURES_ML}

        # Construcción inicial del DataFrame
        input_data = pd.DataFrame([input_row], columns=FEATURES_ML)
        
        # --- CORRECCIÓN CRUCIAL PARA XGBOOST Y COHERENCIA DE TIPOS ---
        # Convertimos la columna al tipo categórico exacto que espera XGBoost
        if "orbit_uncertainity" in input_data.columns:
            input_data["orbit_uncertainity"] = input_data["orbit_uncertainity"].astype(TIPO_ORDINAL_XGB)

        # Prediction
        prediction = model.predict(input_data)[0]

        # Manejo de umbral por si el modelo devuelve probabilidades o clases directas
        pred_idx = 0 if prediction < 0.5 else 1
        clase_predicha = CLASES_TARGET[pred_idx]
        
        probabilidades = None

        if hasattr(model, "predict_proba"):
            prob_arr = model.predict_proba(input_data)[0]
            probabilidades = [float(p) for p in prob_arr]
            
        return {
            "status": "ok",
            "modelo_utilizado": selected_model_name,
            "prediccion": {
                "clase_index": pred_idx,
                "clase_nombre": clase_predicha,
                "probabilidades": probabilidades
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en inferencia: {str(e)}")

if __name__ == '__main__':
    uvicorn.run('app:app', host='0.0.0.0', port=8000, reload=True)
