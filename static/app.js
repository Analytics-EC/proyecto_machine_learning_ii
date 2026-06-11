const { createApp, ref, onMounted, computed } = Vue;
const cuetifyInstance = Vuetify.createVuetify();

createApp({
    setup() {
        // Métricas y estructura general
        const mlStats = ref(null);
        const listaFeaturesNombre = ref([]);
        const metadataFeatures = ref({});

        // Tabla de registros paginados
        const tableRows = ref([]);
        const tableTotal = ref(0);
        const tablePerPage = ref(10);
        const tableLoading = ref(false);
        const tableHeaders = [
            { title: 'Magnitud Absoluta', key: 'absolute_magnitude', sortable: false },
            { title: 'Velocidad Relativa (Kilómetros por Hora)', key: 'relative_velocity_km_per_hr', sortable: false },
            { title: 'Distancia de Aproximación Mínima (Kilómetros)', key: 'miss_dist_kilometers', sortable: false },
            { title: 'Incertidumbre Orbital', key: 'orbit_uncertainity', sortable: false },
            { title: 'Distancia Mínima de Intersección Orbital', key: 'minimum_orbit_intersection', sortable: false },
            { title: 'Excentricidad', key: 'eccentricity', sortable: false },
            { title: 'Semieje Mayor', key: 'semi_major_axis', sortable: false },
            { title: 'Inclinación (Grados)', key: 'inclination', sortable: false },
            { title: 'Longitud del Nodo Ascendente', key: 'asc_node_longitude', sortable: false },
            { title: 'Distancia al Perihelio', key: 'perihelion_distance', sortable: false },
            { title: 'Argumento del Perihelio', key: 'perihelion_arg', sortable: false },
            { title: 'Anomalía Media', key: 'mean_anomaly', sortable: false },
            { title: '¿Es una Órbita Coplanar?', key: 'is_coplanar', sortable: false },
            { title: '¿Es Potencialmente Peligroso?', key: 'hazardous', sortable: false }
        ];

        // Pipeline XAI
        const modeloSeleccionadoXAI = ref('Random Forest');
        const featureSeleccionadaXAI = ref('absolute_magnitude');

        // Formulario de predicción individual
        const formularioPredict = ref({
            model_name: 'Random Forest',
            is_coplanar: false
        });
        const prediccionLoading = ref(false);
        const resultadoPrediccion = ref(null);

        const comandoCurlComputed = computed(() => {
            // Construimos el payload exacto extrayendo los valores de los sliders (.value)
            const payload = {
                model_name: formularioPredict.value.model_name,
                absolute_magnitude: parseFloat(formularioPredict.value.absolute_magnitude || 0),
                relative_velocity_km_per_hr: parseFloat(formularioPredict.value.relative_velocity_km_per_hr || 0),
                miss_dist_kilometers: parseFloat(formularioPredict.value.miss_dist_kilometers || 0),
                orbit_uncertainity: parseFloat(formularioPredict.value.orbit_uncertainity || 0),
                minimum_orbit_intersection: parseFloat(formularioPredict.value.minimum_orbit_intersection || 0),
                eccentricity: parseFloat(formularioPredict.value.eccentricity || 0),
                semi_major_axis: parseFloat(formularioPredict.value.semi_major_axis || 0),
                inclination: parseFloat(formularioPredict.value.inclination || 0),
                asc_node_longitude: parseFloat(formularioPredict.value.asc_node_longitude || 0),
                perihelion_distance: parseFloat(formularioPredict.value.perihelion_distance || 0),
                perihelion_arg: parseFloat(formularioPredict.value.perihelion_arg || 0),
                mean_anomaly: parseFloat(formularioPredict.value.mean_anomaly || 0),
                is_coplanar: formularioPredict.value.is_coplanar ? 1.0 : 0.0
            };

            return `curl -X 'POST' \\\n  '${window.location.origin}/api/predict' \\\n  -H 'Content-Type: application/json' \\\n  -d '${JSON.stringify(payload, null, 2)}'`;
        });

        // --- NUEVO MÉTODO: COPIAR AL PORTAPAPELES ---
        const copiarCurlComando = () => {
            navigator.clipboard.writeText(comandoCurlComputed.value);
            alert("¡Comando cURL copiado al portapapeles con éxito!");
        };

        // Computados dinámicos para la caja de respuesta
        const prediccionCajaColor = computed(() => {
            if (prediccionLoading.value) return '#FAF8F6';
            if (!resultadoPrediccion.value) return '#FFFFFF';

            const probs = resultadoPrediccion.value.probabilidades;
            const esPeligroso = probs ? probs[1] > 0.5 : resultadoPrediccion.value.clase_index === 1;
            return esPeligroso ? '#FDF2F2' : '#F2FDF5';
        });

        const prediccionImagen = computed(() => {
            if (!resultadoPrediccion.value) return '';
            const probs = resultadoPrediccion.value.probabilidades;
            const esPeligroso = probs ? probs[1] > 0.5 : resultadoPrediccion.value.clase_index === 1;
            return esPeligroso ? '/static/hazardous.jpeg' : '/static/safe.jpg';
        });

        const cargarTablaDataset = async ({ page, itemsPerPage }) => {
            tableLoading.value = true;
            try {
                const res = await fetch(`/api/dataset-table?page=${page}&per_page=${itemsPerPage}`);
                const data = await res.json();

                if (data.status === 'ok') {
                    tableRows.value = data.rows;
                    tableTotal.value = data.total;
                }
            } catch (e) {
                console.error("Error cargando tabla:", e);
            } finally {
                tableLoading.value = false;
            }
        };

        // Carga de inicialización de la API (Métricas y sliders)
        const inicializarControlesLanding = async () => {
            try {
                const resStats = await fetch('/api/ml-results');
                mlStats.value = await resStats.json();

                const resMeta = await fetch('/api/dataset-stats');
                const dataMeta = await resMeta.json();
                if (dataMeta.status === 'ok') {
                    metadataFeatures.value = dataMeta.features;
                    listaFeaturesNombre.value = Object.keys(dataMeta.features);
                    if (listaFeaturesNombre.value.length > 0) {
                        featureSeleccionadaXAI.value = listaFeaturesNombre.value[0];
                    }

                    Object.keys(dataMeta.features).forEach(feat => {
                        formularioPredict.value[feat] = dataMeta.features[feat].default;
                    });
                }
            } catch (e) {
                console.error("Error en inicialización:", e);
            }
        };

        // Lógica de consumo de Endpoints XAI y render de gráficos con Plotly
        const calcularPipelineXAI = async () => {
            const model = modeloSeleccionadoXAI.value;

            try {
                const res = await fetch('/api/xai/permutation', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model })
                });
                const data = await res.json();
                if (data.status === 'ok') {
                    const feats = data.permutations.map(p => p.feature);
                    const means = data.permutations.map(p => p.importance_mean);
                    Plotly.newPlot('chart-permutation', [{
                        x: means.reverse(),
                        y: feats.reverse(),
                        type: 'bar',
                        orientation: 'h',
                        marker: { color: '#132B4F' }
                    }], {
                        title: `Importancia por Permutación (${model})`,
                        margin: { l: 160, r: 20, t: 40, b: 40 },
                        font: { size: 11 }
                    }, { responsive: true });
                }
            } catch (e) { }

            await cargarGraficoPDP();

            try {
                const res = await fetch('/api/xai/shap', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model })
                });
                const data = await res.json();
                if (data.status === 'ok') {
                    const absShapMeans = data.features.map((feat, fIdx) => {
                        let total = 0;
                        data.shap_values.forEach(row => { total += Math.abs(row[fIdx] || 0); });
                        return { feature: feat, mean_shap: total / data.shap_values.length };
                    });
                    absShapMeans.sort((a, b) => a.mean_shap - b.mean_shap);

                    Plotly.newPlot('chart-shap', [{
                        x: absShapMeans.map(i => i.mean_shap),
                        y: absShapMeans.map(i => i.feature),
                        type: 'bar',
                        orientation: 'h',
                        marker: { color: '#8B1D2F' }
                    }], {
                        title: `Impacto Medio SHAP en Magnitud de Predicción (${model})`,
                        margin: { l: 160, r: 20, t: 40, b: 40 }
                    }, { responsive: true });
                }
            } catch (e) { }
        };

        const cargarGraficoPDP = async () => {
            try {
                const res = await fetch('/api/xai/pdp-ice', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: modeloSeleccionadoXAI.value,
                        feature: featureSeleccionadaXAI.value
                    })
                });
                const data = await res.json();
                if (data.status === 'ok') {
                    const traces = [];
                    data.ice.forEach((line, idx) => {
                        traces.push({
                            x: data.grid,
                            y: line,
                            type: 'scatter',
                            mode: 'lines',
                            line: { color: 'rgba(150,150,150,0.15)', width: 1 },
                            showlegend: false
                        });
                    });
                    traces.push({
                        x: data.grid,
                        y: data.pdp,
                        type: 'scatter',
                        mode: 'lines+markers',
                        name: 'Curva Promedio PDP',
                        line: { color: '#132B4F', width: 3.5 }
                    });

                    Plotly.newPlot('chart-pdp', traces, {
                        title: `Curvas PDP & ICE: ${featureSeleccionadaXAI.value}`,
                        xaxis: { title: featureSeleccionadaXAI.value },
                        yaxis: { title: 'Respuesta Parcial del Modelo' },
                        showlegend: true,
                        legend: { orientation: 'h', y: -0.2 }
                    }, { responsive: true });
                }
            } catch (e) { }
        };

        const ejecutarPrediccionIndividual = async () => {
            prediccionLoading.value = true;
            resultadoPrediccion.value = null;

            await new Promise(resolve => setTimeout(resolve, 1200));

            try {
                const res = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(formularioPredict.value)
                });
                const data = await res.json();

                if (data.status === 'ok') {
                    resultadoPrediccion.value = {
                        clase_predicha: data.prediccion.clase_nombre,
                        probabilidades: data.prediccion.probabilidades,
                        explicacion_local: data.explicacion_local // CAPTURAMOS XAI
                    };
                    console.log("Predicción asignada con éxito:", resultadoPrediccion.value);
                    
                    // Renderizamos la gráfica después de que Vue actualice el DOM
                    setTimeout(() => {
                        renderLocalShap(data.explicacion_local, data.modelo_utilizado);
                    }, 100);

                } else {
                    console.error("El backend respondió con un estado de error:", data);
                }
            } catch (e) {
                console.error("Error en predicción:", e);
            } finally {
                prediccionLoading.value = false;
            }
        };

        const renderLocalShap = (explicacion, modelName) => {
            const localShapFeatures = explicacion.features;
            const localShapValues = explicacion.shap_values;
            const baseValue = explicacion.base_value;

            // Creamos un array de objetos para poder ordenarlos por magnitud de impacto
            let shapData = localShapFeatures.map((feat, idx) => ({
                feature: feat,
                value: localShapValues[idx]
            }));

            // Ordenamos por valor absoluto para que las variables más influyentes queden visualmente destacadas
            shapData.sort((a, b) => Math.abs(a.value) - Math.abs(b.value));

            const xData = shapData.map(d => d.value);
            const yData = shapData.map(d => d.feature);
            
            // Colores: Guinda si aumenta el riesgo, Navy si lo disminuye
            const colors = shapData.map(d => d.value > 0 ? '#8B1D2F' : '#132B4F'); 

            // Textos dinámicos dependiendo de la matemática interna del modelo
            const unitType = modelName.includes('XGBoost') ? 'Log-Odds' : 'Probabilidad (%)';

            Plotly.newPlot('chart-local-shap', [{
                x: xData,
                y: yData,
                type: 'bar',
                orientation: 'h',
                marker: { color: colors },
                text: xData.map(v => (v > 0 ? '+' : '') + v.toFixed(3)),
                textposition: 'auto',
                insidetextfont: { color: 'white' }
            }], {
                title: `Fuerzas Predictivas SHAP (${unitType} | Base: ${baseValue.toFixed(3)})`,
                margin: { l: 180, r: 20, t: 40, b: 40 },
                xaxis: { title: `Impacto direccional en la predicción` }
            }, { responsive: true });
        };

        onMounted(async () => {
            await inicializarControlesLanding();
            await calcularPipelineXAI();
        });

        const slidersContinuos = computed(() => {
            if (!metadataFeatures.value) return [];
            return Object.keys(metadataFeatures.value)
                .filter(key => {
                    const k = key.toLowerCase();
                    return k !== 'is_coplanar' && !k.includes('uncertainity') && !k.includes('uncertainty');
                })
                .map(key => ({
                    key: key,
                    label: key.replace(/_/g, ' '),
                    min: metadataFeatures.value[key].min,
                    max: metadataFeatures.value[key].max
                }));
        });

        const sliderIncertidumbre = computed(() => {
            if (!metadataFeatures.value) return null;
            const targetKey = Object.keys(metadataFeatures.value).find(key => {
                const k = key.toLowerCase();
                return k.includes('uncertainity') || k.includes('uncertainty');
            });

            if (!targetKey) return null;

            return {
                key: targetKey,
                label: 'Incertidumbre Orbital',
                min: metadataFeatures.value[targetKey].min,
                max: metadataFeatures.value[targetKey].max
            };
        });

        // --- EN EL RETURN AGREGAMOS LAS DOS NUEVAS VARIABLES PARA EL HTML ---
        return {
            mlStats, tableRows, tableTotal, tablePerPage, tableLoading, tableHeaders,
            modeloSeleccionadoXAI, featureSeleccionadaXAI, listaFeaturesNombre, metadataFeatures,
            formularioPredict, prediccionLoading, resultadoPrediccion, prediccionCajaColor, prediccionImagen,
            cargarTablaDataset, calcularPipelineXAI, cargarGraficoPDP, slidersContinuos,
            sliderIncertidumbre, ejecutarPrediccionIndividual,
            comandoCurlComputed, copiarCurlComando // <--- Listas para la vista
        };
    }
}).use(cuetifyInstance).mount('#app');
