# Assignment 1 Report

> ✍️ **Lucía:** las líneas que empiezan por ✍️ son guías para ti — bórralas
> cuando escribas cada sección. Las tablas y cifras salen directamente de
> `uv run python main.py train` (y de `artifacts/metrics.json`); el análisis
> y las conclusiones tienen que estar escritos por ti, con tus palabras.

- **Name**:
- **Student ID**:
- **Email**:
- **Group**: [BBADBA 5A | BBADBA 5B | PPLEDBA 5A | BDBA 3A]

## Dataset

**Facts:**
- Source: CDC Behavioral Risk Factor Surveillance System (BRFSS) 2022, as
  cleaned and published on Kaggle by Kamil Pytlak
  ("Indicators of Heart Disease (2022 UPDATE)", `heart_2022_with_nans.csv`,
  445,132 rows × 40 columns).
- Subset: respondents from **New York State** only — 17,800 rows × 39
  columns (`data/brfss_2022_new_york.csv`, built by
  `data/make_ny_subset.py`).
- One row = one adult who answered the 2022 telephone health survey.
- After defining the target: **17,240 rows, 12.7% with diagnosed diabetes**.

✍️ *Explica por qué elegiste este dataset (te interesa medicina/farmacia),
por qué un solo estado y no todo EE. UU. (tamaño + encaja con un
departamento de salud estatal), y qué significa que sea una encuesta
autodeclarada.*

## Business / real-life framing

**Facts used by the pipeline (`features.py`):**
- Decision supported: which residents get a free glucose (HbA1c) test invitation.
- Budget: tests for **10%** of the population → `CAPACITY = 0.10`.
- Cost of a wasted test (person without diabetes): **€30**.
- Cost of a missed case: **€1,000**.

**Decisions this framing drove:**

| Decision | What the pipeline does |
|---|---|
| Target definition | `HadDiabetes` = "Yes" → 1, "No" → 0. Dropped: pre-diabetes (402), gestational-only (110), no answer (48). |
| Columns removed (reverse causation) | `LastCheckupTime`, `ChestScan`, `FluVaxLast12`, `PneumoVaxEver`, `TetanusLast10Tdap`, `HIVTesting`, `AlcoholDrinkers` |
| Split | Stratified random 80/20 (13,792 / 3,448), seed 42. No time-based split. |
| Tuning | 5-fold stratified CV inside the training set only. |
| Threshold | Top 10% of out-of-fold training scores → **0.323**. The test set is never used to choose it. |
| Ranking metric | PR-AUC (for choosing C) and precision at the 10% budget. |

Cost-only threshold, ignoring the budget: 0.041, which would invite **66%** of
people. The budget, not the cost ratio, is the binding constraint.

✍️ *Escribe con tus palabras:*
- *¿Por qué quitar prediabetes y diabetes gestacional del target?*
- *¿Por qué esas columnas son "consecuencias" del diagnóstico y no causas?
  (p. ej.: los diabéticos van al médico cada año; muchos dejan el alcohol tras
  el diagnóstico.) ¿Por qué eso es justo lo contrario de lo que necesita un
  cribado?*
- *¿Por qué un split aleatorio es válido aquí (un solo año de encuesta, no hay
  "futuro" que filtrar) y por qué estratificado (12,7% de positivos)?*
- *¿Por qué el umbral lo fija el presupuesto y no el ratio de costes? (Con
  €1.000 vs €30, compensaría hacer la prueba al 66% de la gente; no hay dinero
  para eso.)*
- *¿Por qué precisión en el top 10% y no accuracy? (Predecir "nadie es
  diabético" da 87% de accuracy.)*
- *Limitación clave: entrenamos con diabéticos **ya diagnosticados** para
  encontrar a gente **parecida sin diagnosticar**.*

## Data preparation & feature engineering

**What was built (`features.py`):**

| Feature / step | Rule |
|---|---|
| `age`, `age_sq` | Age band → midpoint (18–24 → 21 … 80+ → 82); `(age − 50)²` |
| `bmi`, `bmi_sq`, `obese` | BMI recomputed from weight/height when missing; outside 14–70 → missing; `(bmi − 27)²`; BMI ≥ 30 flag |
| `general_health`, `removed_teeth` | Ordinal answers → 1–5 and 0–3 |
| `sleep_hours`, `short_sleep`, `long_sleep` | Outside 3–14 h → missing; < 6 h and > 9 h flags |
| `frequent_physical_distress`, `frequent_mental_distress` | ≥ 14 bad days out of 30 (CDC definition) |
| `cardiovascular_history` | Any of heart attack / angina / stroke |
| Comorbidities, difficulties, physical activity | Yes/No → 1/0 |
| Sex, race/ethnicity, smoking, e-cigarettes | One-hot. Missing = own category. Reference levels: Female, White non-Hispanic, Never smoked, Never used e-cigarettes |
| Missing numeric values | Training-set median + a "was missing" indicator (base columns only) |
| Scaling | StandardScaler, fitted on train only |
| Result | **67 model inputs** |

**Removed after testing:**
- Count of comorbidities and count of difficulties.
- Number of unanswered questions.

**Ablation (5-fold CV on the training set, scikit-learn, C = 1):**

| Feature set | ROC-AUC | PR-AUC |
|---|---|---|
| Raw columns (one-hot, no engineering) | 0.8176 ± 0.0097 | 0.3845 ± 0.0051 |
| **Engineered features (used)** | **0.8191 ± 0.0092** | **0.3895 ± 0.0060** |
| Engineered, minus non-linear terms | 0.8160 ± 0.0102 | 0.3803 ± 0.0093 |
| Raw + consequence columns (leaky, not used) | 0.8272 ± 0.0082 | 0.4025 ± 0.0083 |

**Data-quality findings:**
- **Dummy-variable trap.** With full one-hot encoding, PyTorch and sklearn
  weights differed by up to 0.27 while their predictions matched.
  - Cause: each group of dummies sums to 1, which duplicates the intercept.
  - Fix: drop one reference level per category.
- **Duplicate missing indicators.** `age`/`age_sq` and `bmi`/`bmi_sq`
  (and others) produced identical "was missing" columns.
  - Fix: indicators only for base columns.
- **Collinear squared terms.** `age` and `age²` are ~0.99 correlated, and
  gradient descent crawled along that direction.
  - Fix: centre before squaring.
  - Result: the PyTorch loops now reach sklearn's optimum in ~500 epochs
    instead of not reaching it in 3,000.

✍️ *Aquí está el 30% de la nota. Para cada feature explica **por qué** existe.
Pista: una regresión logística solo dibuja relaciones lineales, así que los
cuadrados, umbrales y "máximos" existen para captar curvas (edad, IMC, sueño
en U) que el modelo no puede construir solo. Y explica por qué quitaste las
sumas: un modelo lineal ya puede sumar columnas él solo.*

✍️ *Sé honesta con la tabla de ablación: la mejora del feature engineering es
pequeña (+0,0015 ROC-AUC, +0,005 PR-AUC), casi dentro del ruido de la
validación cruzada. ¿Por qué crees que pasa? (Pista: las columnas originales
ya son casi todas categorías, y al pasarlas a one-hot cada nivel recibe su
propio peso.) Y explica por qué el conjunto "leaky" puntúa más y aun así es
peor modelo.*

## Modeling: three implementations, one model

**Model:**
- L2-regularised logistic regression, because the target is binary.
- All three implementations minimise the same objective:
  `mean(BCE) + (λ/2)·‖w‖²`, with `λ = 1/(C·n)`. This is equivalent to
  sklearn's `C`, and the bias is not penalised.
- **C = 0.1:** 5-fold CV on PR-AUC (0.001 → 0.367, 0.01 → 0.383, 0.1 → 0.389,
  1 → 0.389, 10 → 0.390). Chosen as the smallest C within 0.001 of the best.
- **Learning rate = 2.0:** full-batch gradient descent, 300-epoch check
  (0.01 → 0.412, 0.1 → 0.313, 0.5 → 0.301, 1 → 0.299, 2 → 0.2987,
  4 → 0.362, which oscillates). Then 3,000 epochs.
- **Manual PyTorch:** raw tensors, autograd, hand-written
  `w -= lr * w.grad`, BCE written by hand as `softplus(z) − y·z`.
- **Standard PyTorch:** `nn.Linear` + `BCEWithLogitsLoss` + `optim.SGD`,
  with `weight_decay = λ` on the weights only.

**Test set (3,448 people, 437 with diabetes). Every row invites ~10%:**

| Model | ROC-AUC | PR-AUC | Log loss | Invited | Diabetics found | Precision | Recall | Cost (€) |
|---|---|---|---|---|---|---|---|---|
| Baseline: random invitations | 0.490 | 0.124 | 0.380 | 345 | 41 | 11.9% | 9.4% | 405,120 |
| Baseline: oldest first | 0.671 | 0.182 | – | 345 | 56 | 16.2% | 12.8% | 389,670 |
| scikit-learn | 0.808 | 0.385 | 0.309 | 337 | 139 | 41.2% | 31.8% | 303,940 |
| PyTorch manual | 0.808 | 0.385 | 0.309 | 337 | 139 | 41.2% | 31.8% | 303,940 |
| PyTorch nn.Module | 0.808 | 0.385 | 0.309 | 337 | 139 | 41.2% | 31.8% | 303,940 |

**95% bootstrap intervals (scikit-learn, 1,000 resamples):**
- ROC-AUC: 0.789–0.828
- Precision at the budget: 36.1%–46.6%

**Agreement with scikit-learn:**

| | Max \|Δ prob\| | Mean \|Δ prob\| | Max \|Δ weight\| | Same invite decision | Final objective |
|---|---|---|---|---|---|
| PyTorch manual | 1.7e-3 | 6.2e-5 | 5.3e-3 | 3,448 / 3,448 | 0.298620 |
| PyTorch nn.Module | 1.7e-3 | 6.2e-5 | 5.3e-3 | 3,448 / 3,448 | 0.298620 |
| scikit-learn (L-BFGS) | – | – | – | – | 0.298620 |

**Audit by group (test set, at the budget threshold):**

| Group | N | Diabetes rate | Share invited | Recall | ROC-AUC |
|---|---|---|---|---|---|
| Female | 1,833 | 11.5% | 8.2% | 30.0% | 0.807 |
| Male | 1,615 | 14.1% | 11.5% | 33.5% | 0.808 |
| White, non-Hispanic | 2,169 | 11.3% | 8.1% | 28.2% | 0.801 |
| Black, non-Hispanic | 363 | 19.6% | 20.9% | 43.7% | 0.774 |
| Hispanic | 514 | 13.2% | 10.5% | 38.2% | 0.823 |

✍️ *Explica: ¿coinciden los tres? (Sí: misma función objetivo hasta la 6ª
cifra decimal y las mismas 3.448 decisiones.) ¿Por qué las pequeñas
diferencias que quedan? (sklearn usa L-BFGS en float64; PyTorch usa descenso
de gradiente en float32.) Cuenta la historia de cómo NO coincidían al
principio (la trampa de las dummies) y cómo lo investigaste. Eso es
exactamente lo que pide la rúbrica.*

✍️ *Traduce los resultados a negocio: con el mismo presupuesto el modelo
encuentra 139 diabéticos frente a 41 al azar (3,4 veces más) y ahorra
~€100.000 en el conjunto de test.*

✍️ *Comenta la auditoría: ¿por qué decidiste incluir sexo y raza (en medicina
hay diferencias biológicas reales y las guías de diabetes usan la etnia), a
diferencia de lo que harías en un problema de contratación? ¿Qué te dice que
el modelo invite proporcionalmente más a personas negras (mayor prevalencia)
y tenga algo menos de AUC en ese grupo?*

## Limitations & next steps

✍️ *Ideas que salen de los datos (escríbelas tú y añade cómo lo arreglarías):*
- *El target es "diabetes diagnosticada", no "diabetes": los no
  diagnosticados están etiquetados como 0.*
- *Encuesta autodeclarada (peso, altura y enfermedades dichas por el propio
  encuestado).*
- *Una sola foto en el tiempo (2022): no se puede validar en datos
  posteriores.*
- *Solo Nueva York: ¿se generaliza a otros estados? (Prueba posible: evaluar
  con los datos de otro estado.)*
- *Test pequeño en positivos (437): intervalos de ±5 puntos en precisión.*
- *El modelo lineal ya está cerca de su techo (~0,82 de AUC): un modelo no
  lineal (árboles) podría captar interacciones.*
- *Los costes de €30 y €1.000 son supuestos: el dashboard deja cambiarlos.*

## Generative AI use disclosure

✍️ *Revisa y ajusta este borrador para que refleje lo que de verdad hiciste:*

I used Claude (Anthropic) as an assistant during this assignment:
- It suggested candidate datasets and explained the trade-offs of each. I
  chose the CDC BRFSS / diabetes-screening framing.
- It proposed options for each design decision (target definition, which
  columns to exclude, split, missing-value handling, features, capacity and
  costs), with its recommendations. I reviewed and accepted them.
- It wrote the code in `features.py`, `train.py`, `app.py` and
  `data/make_ny_subset.py`. It also diagnosed the dummy-variable and
  collinearity issues that made the three implementations disagree.
- It filled in the factual tables of this report from the pipeline output.
  The analysis and conclusions in the prose are my own.
