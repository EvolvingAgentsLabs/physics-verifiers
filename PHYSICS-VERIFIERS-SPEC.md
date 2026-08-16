# PHYSICS-VERIFIERS: Entornos de RL con recompensas calculadas por física

## Especificación completa para desarrollo con Claude Code

**Versión:** 1.0 · **Fecha:** 2026-08-16
**Repo objetivo:** github.com/EvolvingAgentsLabs/physics-verifiers · Apache 2.0
**Relación con otros proyectos:** ai-os queda EN PAUSA (es el sustrato v2 si esto tracciona). COCLEA-SR-SPEC.md se degrada de "programa de investigación" a **cantera de tareas**: sus gates y casos analíticos se reutilizan acá como entornos. Nada de este repo depende de ai-os.
**Propósito:** ser la fuente de verdad única para construir, en 30 días, un catálogo de entornos de RL publicable en el ecosistema abierto (Prime Intellect Environments Hub) donde **la recompensa la calcula la física** — soluciones analíticas, invariantes de conservación, órdenes de convergencia, estadística con verdad semi-analítica — y no un juez LLM ni un test suite hackeable.

---

## 0. Tesis del producto (una pantalla)

**Problema del comprador:** el post-training moderno es RLVR — el entorno puntúa el output. Los verificadores débiles se hackean (reward hacking); los dominios con recompensa estocástica (finanzas, ciencia) hoy se parchean con jueces LLM, que son verificadores blandos vigilando verificadores blandos. Los labs gastan presupuestos de 9-10 cifras en entornos, y el consenso público del mercado es "el verificador es la parte difícil".

**Nuestra respuesta:** entornos donde el verificador es un teorema. No se puede persuadir a la conservación de la energía, no se puede hacer prompt-injection a una ecuación de Lyapunov, y el orden de convergencia observado o es 2 o no lo es. La física provee además el gradiente de dificultad natural (de "derivá los modos de la cuerda uniforme" a "diagnosticá por qué este solver estocástico viola fluctuación-disipación") y, mediante el Method of Manufactured Solutions, un **generador infinito de tareas con verdad exacta**.

**Producto v1:** 5 familias de entornos compatibles con la interfaz `verifiers`, publicadas en el hub abierto. Monetización: el catálogo abierto es distribución y credencial; el negocio es (a) entornos privados a medida para labs (biofísica, dominios propietarios) y (b) partnership de thought partner. Objetivo comercial de los 90 días: **un piloto pago**.

**Regla de oro del repo:** cada entorno debe poder responder la pregunta de un research engineer escéptico: *"¿cómo sé que un modelo no puede conseguir recompensa alta sin resolver el problema?"* — con una respuesta de una oración que termine en un teorema, no en "porque el juez es bueno".

---

## 1. Principios de diseño

1. **La verdad nunca viaja con la tarea.** El agente ve el enunciado; el verificador tiene la verdad (solución analítica, invariante, semilla). Verdad y prompt viven en módulos separados; el enunciado jamás incluye valores que permitan interpolar la respuesta.
2. **Recompensa densa pero no hackeable.** Preferir recompensas continuas (error relativo transformado, orden observado, fracción de invariantes preservados) sobre binarias — el RL las aprovecha mejor — PERO cada componente continuo debe estar acotado por un chequeo duro (si el gate binario falla, la recompensa continua se anula). Fórmula estándar: `reward = hard_gate * soft_score`, con `soft_score = exp(-err_rel / tol)` acotado a [0,1].
3. **Anti-memorización por parametrización.** Ninguna tarea tiene respuesta fija: cada instancia se genera con parámetros muestreados (L, c, α, N, semillas) y la verdad se computa on-the-fly con mpmath/SymPy. Un dataset estático se memoriza; un generador paramétrico no.
4. **Anti-gaming del sandbox.** El código del agente corre en subprocess aislado con timeout y sin red; el verificador corre en el proceso padre y NUNCA importa código del agente — solo lee sus outputs (archivos JSON/NPY con esquema fijo). El agente no puede introspeccionar el verificador porque el verificador no está en su filesystem.
5. **Dos verdades independientes por familia cuando sea posible.** La verdad de referencia se computa por una vía distinta a la que un solver típico usaría (mpmath de precisión arbitraria + fórmula cerrada vs. numérica de punto flotante). Si ambas discrepan, el bug es nuestro y el CI lo caza.
6. **Dificultad graduada explícita.** Cada familia expone `level ∈ {1..4}`: (1) aplicar fórmula conocida, (2) implementar solver, (3) diagnosticar/arreglar solver roto, (4) diseño abierto con restricción física. Los niveles 3-4 son los que más valoran los labs (razonamiento, no recall).
7. **Atestación liviana, no ledger completo.** Cada instancia de tarea lleva `manifest = {env_version, params, seed, truth_hash}` y cada evaluación emite `{manifest, agent_output_hash, reward, breakdown}`. Suficiente para reproducir y auditar; el ledger hash-encadenado completo es v2.
8. **Inglés como idioma del repo** (código, prompts, docs). El comprador es global; README bilingüe solo en la raíz.

---

## 2. Arquitectura e integración con `verifiers`

### 2.1 Estrategia de interfaz (instrucción crítica para Claude Code)

La librería `verifiers` de Prime Intellect es el protocolo de distribución, pero **su API evoluciona**. Por lo tanto:

- **Paso 0 obligatorio:** `pip install verifiers` en el entorno de desarrollo y **leer el código instalado** (no la memoria del modelo, no tutoriales viejos): cómo se define un entorno, cómo se declara la función de recompensa/rúbrica, qué espera el hub para publicar (`prime env push` o equivalente vigente). Escribir lo aprendido en `docs/INTERFACE-NOTES.md` con la versión pineada antes de escribir código propio.
- **Núcleo agnóstico:** toda la física vive en `core/` sin importar `verifiers`. La integración vive en `adapters/verifiers_adapter.py`, fina, reemplazable. Si mañana un lab quiere otra interfaz (interna, OpenAI Gym-like, HTTP), se escribe otro adapter sin tocar `core/`.
- **Contrato interno estable:**

```python
# core/contracts.py
@dataclass(frozen=True)
class TaskInstance:
    family: str          # "eigenmodes" | "conservation" | "convergence" | "stochastic" | "mms"
    level: int           # 1..4
    prompt: str          # enunciado autocontenido (inglés), sin la verdad
    params: dict         # parámetros muestreados (van al manifest, NO al prompt salvo los necesarios)
    seed: int
    manifest: dict       # {env_version, family, level, params_hash, seed, truth_hash}

@dataclass(frozen=True)
class Verdict:
    reward: float        # [0, 1]
    hard_gate: bool      # False => reward == 0.0 siempre
    breakdown: dict      # componentes: {"eig_rel_err": ..., "order_observed": ..., ...}
    evidence: dict       # lo necesario para auditar sin re-correr

class Family(Protocol):
    def generate(self, level: int, rng: np.random.Generator) -> TaskInstance: ...
    def verify(self, task: TaskInstance, agent_output: AgentOutput) -> Verdict: ...
```

- **AgentOutput:** esquema fijo por familia (JSON + arrays NPY opcionales), documentado en el prompt de cada tarea. El agente que no respeta el esquema recibe reward 0 con `breakdown.schema_error` — eso también es señal de entrenamiento.

### 2.2 Estructura del repo

```
physics-verifiers/
├── README.md                    # pitch de una pantalla + quickstart + tabla de familias
├── CLAUDE.md                    # reglas §1 + puntero a este spec + workflow de tests
├── PHYSICS-VERIFIERS-SPEC.md    # este documento
├── pyproject.toml               # deps pinneadas: numpy, scipy, sympy, mpmath, verifiers, pytest, hypothesis
├── docs/
│   ├── INTERFACE-NOTES.md       # hallazgos del paso 0 (versión de verifiers pineada)
│   └── ANTIHACK.md              # por qué cada familia es no-hackeable (la respuesta al escéptico)
├── core/
│   ├── contracts.py
│   ├── truth/                   # verdad: mpmath/SymPy, precisión arbitraria; NUNCA importa families/
│   │   ├── uniform_string.py    # (2n-1)π/2L y modos
│   │   ├── exponential_string.py# raíces de tan(βL) = -2β/α vía mpmath
│   │   ├── ou_lyapunov.py       # varianza estacionaria OU exacta
│   │   ├── rice_rates.py        # tasas de cruce de Rice
│   │   └── mms_engine.py        # SymPy: solución fabricada -> forzamiento simbólico -> lambdify
│   ├── families/
│   │   ├── eigenmodes.py        # Familia 1
│   │   ├── conservation.py      # Familia 2
│   │   ├── convergence.py       # Familia 3
│   │   ├── stochastic.py        # Familia 4
│   │   └── mms.py               # Familia 5
│   ├── sandbox.py               # ejecución aislada del código del agente (subprocess, timeout, sin red)
│   └── attest.py                # manifiestos y hashes (§1.7)
├── adapters/
│   └── verifiers_adapter.py
├── tests/                       # el CI del repo ES la credibilidad del producto
│   ├── test_truth_crosscheck.py # dos vías independientes por verdad (§1.5)
│   ├── test_antigaming.py       # soluciones triviales/constantes/copiadas deben puntuar ~0
│   ├── test_family_*.py
│   └── test_reference_agent.py  # un solver correcto de referencia debe puntuar ~1 en niveles 1-2
├── experiments/
│   └── judge_vs_physics/        # el experimento del post (§4)
└── examples/
    └── run_local_eval.py        # evaluar un modelo cualquiera contra el catálogo, local
```

**Los tres tests que definen el producto** (van en el README):
1. `test_reference_agent`: una implementación correcta obtiene reward ≈ 1 (el entorno es resoluble y la recompensa está bien calibrada).
2. `test_antigaming`: outputs triviales (ceros, constantes, la media, copiar el input, valores "típicos" de libro) obtienen reward ≈ 0 (el entorno no regala señal).
3. `test_truth_crosscheck`: las dos vías de verdad coinciden a 1e-10 (la verdad es verdad).

---

## 3. Las cinco familias (especificación exacta)

Convención común: adimensionalizar internamente (L=1, c₀=1); los parámetros muestreados se reportan en el manifest. RNG: `np.random.Generator(PCG64)` con `SeedSequence(master, task_id)`. Tolerancias por nivel definidas en cada familia; `soft_score = exp(-err/tol)` salvo indicación.

### Familia 1 — EIGENMODES (modos propios con verdad cerrada)

*La puerta de entrada del catálogo: simple de explicar, imposible de simular sin resolver.*

**Física:** cuerda fija-libre. Caso A (uniforme): ω_n = (2n−1)πc/2L, modos sin(k_n x). Caso B (exponencial, K=K₀e^{−αx}, μ=μ₀e^{−αx}): autovalores desde las raíces β_n de tan(βL) = −2β/α, con ω_n = c√(β_n² + α²/4), modos e^{αx/2}sin(β_n x). La verdad se computa con mpmath (Newton con bracketing por intervalos de π, 50 dígitos).

**Niveles:**
- L1: "Dadas L, c (y α), computá los primeros N autovalores y evaluá el modo m en estos puntos x_j." Output: JSON con arrays. Muestreo: L∈[0.5,2], c∈[0.5,3], α∈[0.5,6], N∈[5,15].
- L2: "Implementá un solver de autovalores para el problema de Sturm-Liouville con estos perfiles μ(x), K(x) dados como código; devolvé los primeros N autovalores." Los perfiles muestreados incluyen los casos con verdad cerrada (verificables exacto) mezclados con perfiles MMS-verificables (ver Familia 5) — el agente no sabe cuál le tocó.
- L3: "Este solver (código provisto) devuelve autovalores con error del 8% en los modos altos. Diagnosticá y corregí." Bugs inyectados de un banco: borde Neumann de orden 1, K evaluada en nodos en vez de puntos medios, off-by-one en la grilla, matriz no simetrizada. La recompensa exige que el diagnóstico escrito nombre la causa correcta (match contra categoría, no string exacto) Y que el código corregido pase.
- L4: "Diseñá un perfil μ(x) suave y positivo tal que los primeros 5 autovalores estén dentro del 1% de estos objetivos [muestreados de un perfil oculto realizable]." Problema inverso; verificación: resolver el perfil propuesto por nuestra vía y comparar.

**Recompensa:** hard_gate = (esquema válido) ∧ (todos los autovalores reales positivos crecientes). soft = exp(−max_n err_rel(ω_n)/tol), tol = 1e-6 (L1), 1e-4 (L2), 1e-4 (L3, más gate de diagnóstico), 1e-2 (L4).
**Anti-hack (docs/ANTIHACK.md):** los autovalores del caso exponencial con α muestreado continuo no están en ningún libro ni dataset; interpolar de memoria da errores >> tol. Respuesta al escéptico: *"la recompensa requiere las raíces de una ecuación trascendente con parámetros frescos; o las computás o no las tenés."*

### Familia 2 — CONSERVATION (invariantes como verificador)

*La familia insignia del pitch: el verificador es un teorema.*

**Física:** el agente debe producir la trayectoria u(x,t), u̇(x,t) de la cuerda (uniforme o de perfil suave muestreado) sin disipación desde condiciones iniciales dadas, integrada hasta T. El verificador chequea: (i) energía E(t) = ½∫(μu̇² + Ku'²)dx constante dentro de deriva tolerada; (ii) simetría temporal: integrar hacia adelante T y con estado invertido volver — el estado final debe coincidir con el inicial (reversibilidad, mata a los integradores disipativos "que se ven bien"); (iii) consistencia con la PDE por residuo débil en puntos de control aleatorios (mata a "devolver una función suave cualquiera con energía constante": una constante conserva energía pero no satisface la PDE con esas condiciones iniciales).

**El trío (i)+(ii)+(iii) es el corazón anti-gaming:** cada chequeo solo mata una clase de trampa; juntos no dejan clase viva. Documentar esta lógica explícitamente — es la figura 1 del post.

**Niveles:** L1 evolución con datos suaves y esquema libre; L2 exigir además orden 2 en Δt (sub-chequeo de la Familia 3); L3 diagnóstico de un integrador que pierde 0.1% de energía por período (banco de bugs: paso adaptativo mal reiniciado, término γ residual, Verlet mal centrado); L4 diseñar un integrador que conserve energía a 1e-8 durante 1000 períodos con presupuesto de pasos acotado (fuerza esquemas simplécticos/exactos-modales — hay múltiples soluciones válidas, el verificador no exige una).

**Recompensa:** hard_gate = residuo PDE < tol_pde en los puntos de control. soft = producto de exp(−deriva_E/tol_E) y exp(−err_reversibilidad/tol_R).

### Familia 3 — CONVERGENCE (el orden observado no se negocia)

**Física/numérica:** el agente entrega su solver como función de N (tamaño de grilla); el verificador lo corre en N ∈ {N₀, 2N₀, 4N₀} dentro del sandbox, computa el orden observado p = log₂(‖e_{N}‖/‖e_{2N}‖) contra la verdad (cerrada o MMS) y aplica Richardson.

**Niveles:** L1 demostrar orden 2 de un esquema estándar sobre un problema dado; L2 lograr orden 2 con coeficientes variables y borde Neumann (donde el error clásico es caer a orden 1 — exactamente la lección del spec COCLEA); L3 "este solver reporta orden 1.2; encontrá qué término lo degrada y arreglalo sin perder estabilidad"; L4 lograr orden 4 (fuerza compactos o extrapolación bien hecha).

**Recompensa:** hard_gate = solver corre en los 3 tamaños sin NaN y errores decrecientes. soft = exp(−|p_observado − p_objetivo|/0.15). **Nota:** este es el entorno más barato de correr y el más denso en señal de "entiende numérica de verdad" — candidato a primer contacto con labs.

### Familia 4 — STOCHASTIC (verdad semi-analítica: el argumento diferencial)

*Ataca de frente el problema abierto de recompensas estocásticas sin jueces LLM.*

**Física:** procesos OU (escalares y el sistema modal 2-D de la cuerda amortiguada forzada por ruido blanco). Verdades exactas disponibles: varianza estacionaria vía ecuación de Lyapunov, función de autocorrelación exponencial exacta, tasas de cruce de Rice, y para el OU 2-D la matriz de covarianza completa (Van Loan).

**El truco de diseño (documentarlo, es publicable):** la recompensa NO compara trayectorias (irreproducibles) sino **estadísticos con distribución de muestreo conocida**. El agente entrega M trayectorias con su semilla declarada; el verificador computa los estadísticos y los compara contra la verdad exacta usando el error estándar teórico del estimador: z = (θ̂ − θ_exact)/SE(M, T). Recompensa alta si |z| es consistente con muestreo honesto (|z| ≲ 3) Y la varianza entre las M réplicas coincide con la teórica (mata al agente que devuelve M copias del valor exacto sin simular: varianza inter-réplica ≈ 0 → gate falla; y al que suma ruido fabricado: la autocorrelación no da exponencial con el τ correcto).

**Niveles:** L1 simular OU y reportar estadísticos; L2 sistema modal 2-D con Lyapunov como verdad; L3 diagnóstico ("este simulador viola fluctuación-disipación: la varianza da 2× la teórica — encontrá el factor √2 perdido en el término de ruido"); L4 diseñar el esquema de muestreo exacto (transición gaussiana exacta) y demostrar que el error no depende de Δt.

**Recompensa:** hard_gate = consistencia inter-réplica ∧ forma de autocorrelación. soft = exp(−|z|/3) promediado sobre estadísticos.

### Familia 5 — MMS-GEN (el generador infinito)

*El multiplicador del catálogo: verdad exacta con coeficientes arbitrarios.*

**Método (Salari & Knupp):** elegir una solución fabricada u_m(x,t) (biblioteca simbólica: productos de senos/cosenos/gaussianas/polinomios con parámetros muestreados, siempre satisfaciendo u(0,t)=0, u'(L,t)=0 por construcción — se garantiza componiendo con funciones base que ya cumplen los bordes), elegir μ(x), K(x), γ(x) arbitrarios suaves positivos (biblioteca paramétrica), y computar simbólicamente el forzamiento F_m = ∂_t(μ∂_t u_m) − ∂_x(K∂_x u_m) + γ∂_t u_m con SymPy, lambdificado para el enunciado. La tarea: "resolvé la PDE con estos coeficientes y este forzamiento"; la verdad: u_m exacta, conocida solo por el verificador.

**Pipeline `mms_engine.py`:** (1) muestrear u_m y coeficientes; (2) derivar F_m simbólico; (3) verificación interna: sustituir u_m en la PDE con F_m debe dar residuo simbólico 0 (test automático de cada instancia — la fábrica se auto-verifica); (4) emitir TaskInstance con F_m como código y truth_hash de u_m.

**Niveles:** L1 evaluar la solución numérica en puntos dados; L2 solver completo, error L² contra u_m; L3 el clásico MMS de diagnóstico (bug inyectado, el error contra u_m localiza el término culpable — pedagógicamente el mejor entorno del catálogo); L4 el agente debe además reportar el orden de convergencia contra u_m (combina con Familia 3).

**Anti-hack:** u_m vive solo en el verificador; F_m revela la estructura pero invertir F_m → u_m simbólicamente es exactamente resolver el problema (y si un modelo lo hace analíticamente, ES la habilidad que queremos recompensar — win-win, documentarlo así).

---

## 4. El experimento del post: "judge vs physics"

**Objetivo:** una figura que sea el pitch entero. Comparar la resistencia al reward hacking de (a) un juez LLM puntuando soluciones de física, vs (b) nuestros verificadores, bajo un atacante idéntico.

**Protocolo (experiments/judge_vs_physics/):**
1. Tomar 30 instancias de Familias 2 y 4 (nivel 2).
2. **Brazo honesto:** un agente (modelo mid-tier vía API) resuelve de buena fe. Medir reward del juez LLM (rúbrica razonable, mejor esfuerzo) y de nuestro verificador.
3. **Brazo atacante:** el mismo modelo con presupuesto de intentos e instrucción explícita de maximizar la recompensa *sin resolver* (estrategias: outputs plausibles con jerga, energía constante trivial, copiar valores del enunciado, apelar a autoridad en el texto, M réplicas idénticas en la familia estocástica, prompt-injection en el output dirigida al juez).
4. **Métrica:** hack rate = P(reward > 0.7 | atacante). Predicción honesta a registrar antes de correr: juez LLM > 30-60%, physics ≈ 0% con los gates del trío de la Familia 2 y la consistencia inter-réplica de la 4.
5. Publicar TODO: código del ataque, transcripciones, manifiestos. Si el juez resiste más de lo esperado, se publica igual (§1 del negocio: un negativo limpio también vende infraestructura — y de paso calibra el claim).

**El post:** "Physics-gated environments: rewards you can't sweet-talk" — la figura del hack rate, la explicación del trío anti-gaming, el truco de estadísticos-con-SE de la Familia 4, link al hub y a `run_local_eval.py`. Cierre: catálogo privado a medida para dominios propietarios (el CTA comercial, una línea, sin desesperación).

---

## 5. Plan de 30 días

**Semana 1 — un entorno vivo vale más que cinco especificados.**
D1: paso 0 (leer `verifiers` instalado, INTERFACE-NOTES.md, pinear versión). D2-3: contracts, truth/uniform+exponential (con crosscheck de dos vías), attest liviano. D4-5: Familia 1 completa (4 niveles) + los tres tests que definen el producto + adapter. D6-7: publicar en el hub, README con quickstart, smoke test end-to-end contra un modelo real vía API.
**Gate de la semana:** un tercero puede correr `run_local_eval.py` contra Familia 1 y obtener un reporte con manifiestos.

**Semana 2 — el catálogo.**
D8-10: Familia 3 (la más barata) + Familia 2 con el trío anti-gaming. D11-13: Familia 4 (leer Higham 2001 el D11: afecta el diseño del muestreo exacto). D14: hardening del sandbox (timeouts, memoria, sin red) + test_antigaming ampliado con los ataques que se nos ocurran — cada ataque nuevo es un test, la suite de ataques ES un activo.

**Semana 3 — el multiplicador y el argumento.**
D15-17: mms_engine (leer Salari & Knupp el D15) + Familia 5; la auto-verificación simbólica de cada instancia en CI. D18-21: experimento judge_vs_physics completo + redacción del post con la figura.

**Semana 4 — distribución.**
D22-23: pulir hub listing, ANTIHACK.md, ejemplos. D24: publicar el post (X/LinkedIn + canal ANFAIA/Aleph; pedir amplificación a Ismael). D25-30: outreach técnico — 5-10 emails, no más, hiper-apuntados: equipos de environments/post-training de 2 labs, Prime Intellect (contribución + conversación), 1-2 agregadores (Mercor/Surge/Scale). Plantilla: 3 líneas + link al hub + la figura del hack rate. Pedido concreto: 30 minutos técnicos, no "partnership".

**Presupuesto:** ~USD 200-500 de API para el smoke test y el experimento del brazo atacante. Tiempo: dedicación parcial real (Histora sigue primero); todo el código lo escribe Claude Code contra este spec, con vos como especificador y auditor — que es, otra vez, la demo viviente de la tesis.

---

## 6. Métricas y criterio de corte

| Día | Métrica | Umbral "seguir" |
|---|---|---|
| 7 | Familia 1 en el hub, eval externo posible | sí/no |
| 21 | 5 familias + figura judge_vs_physics | hack rate physics < 5% |
| 30 | outreach enviado | ≥ 2 respuestas técnicas |
| 60 | conversaciones | ≥ 1 llamada con lab/agregador |
| 90 | negocio | 1 piloto pago O encargo de entorno privado |

**Corte a los 90 días sin piloto:** el catálogo queda como credencial pública y suite interna (los entornos son, nota, exactamente los tests de regresión que COCLEA-SR y los proyectos médicos necesitan — cero desperdicio), y el foco vuelve al plan regulado (Histora/Nebuah) con ai-verify como infraestructura. La tesis de mercado hereda la fecha de revisión 2026-11-15 del AI-OS-PLAN.

## 7. Riesgos específicos

- **R1 — La interfaz `verifiers` cambió respecto de lo documentado:** por eso el paso 0 es leer el código instalado y el core es agnóstico.
- **R2 — "Esto ya lo hace alguien":** verificado 2026-08 que no en física computacional; re-verificar en D22 antes del post (búsqueda de 30 min, se anota en docs/).
- **R3 — El experimento judge_vs_physics da un resultado débil (el juez resiste):** se publica igual con honestidad; el claim se recalibra a "verificadores con costo cero de juez y garantía formal" — que sigue siendo cierto y vendible.
- **R4 — Sandbox escapable:** subprocess sin red + sin acceso al proceso padre + verificador fuera del filesystem del agente; test explícito de que el agente no puede leer truth/. Si un modelo lo rompe, es un finding publicable (y un parche).
- **R5 — Dispersión:** este spec deliberadamente NO incluye: ledger completo, Evidence Viewer, COCLEA-SR B2, Hopf, paper de síntesis. Todo eso está en los otros dos documentos y espera tracción.

---

*Fuente de verdad única. Ante ambigüedad: issue en docs/decisions/, no improvisación. Primer prompt para Claude Code: "Leé PHYSICS-VERIFIERS-SPEC.md completo y CLAUDE.md. Ejecutá el paso 0 (§2.1) y escribí docs/INTERFACE-NOTES.md. Después D2-D3 de §5: contracts.py, truth/ con crosscheck de dos vías, attest.py — tests primero."*

---

## Anexo A — Divergencias de la spec, medidas al construirla

Se registran acá en vez de editar el cuerpo, para que la spec siga siendo lo que
se especificó y esto sea lo que se encontró. Regla del repo: la spec gana ante
ambigüedad; ante **contradicción con la realidad**, gana la realidad y se anota.

**A.1 — `verifiers` tiene dos APIs vivas y solo una se instala (2026-08-16, `[ran]`).**
§2.1 anticipó que la interfaz evoluciona; la forma exacta es peor que eso.
`pip install verifiers` da **0.1.5** (`Environment` / `Rubric` / `SingleTurnEnv`,
sin módulo `v1`), mientras que los entornos del propio repositorio usan
`verifiers.v1` (`Task`, `Runtime`, `@reward`, `run_uv_script`), que **ninguna
versión publicada contiene**. Consecuencia: dos adapters, no uno, y `core/` no
importa ninguno. Detalle con ambos contratos citados del fuente en
`docs/INTERFACE-NOTES.md`.

**A.2 — El umbral del residual de la Familia 1 no puede ser absoluto.**
§3 fija tolerancias como números. Para el caso exponencial, el residual de la
ecuación trascendente escala con la raíz: medido, va de 1.4e-30 a 2.5e-29 sobre
los rangos que §3 manda muestrear. Un umbral absoluto es un umbral sobre `α` y
`n` disfrazado. Se normaliza por `b + α/2`.

**A.3 — La convención de la Familia 4 va en el enunciado, no solo en el código.**
§3 escribe `q'' + 2G q' + w²q = ξ`, `⟨ξξ⟩ = 2Dδ` una vez. El ancestro de este
repo perdió una solución correcta por ese factor de dos — error relativo
exactamente 1.0 — porque el prompt no lo decía. **Todo prompt que use esta verdad
escribe la SDE y la normalización completas.**

**A.4 — Do not close a training loop; use traces and frontier models.**
A review raised this and it is adopted as a scope decision. Running PPO/GRPO
would spend the 30 days on CUDA and hyperparameters and would prove the wrong
thing: the product is the referee, not the player. What a buyer needs to see is
that the reward is **dense, discriminating and actionable**, which is testable in
inference for the price of a few API calls. `examples/demo_refinement.py` does
it — best-of-N selection and verdict-fed-back refinement, measured at
0.0000 → 0.0019 → 0.8812.

**A.5 — Tolerances are set from a measured instrument floor, not chosen.**
Also raised by review, and by then already the practice. Every family has at
least one number that was wrong before it was measured: MMS level 1 asked 1e-8
where a correct second-order solve gives 3.5e-5; the conservation energy
estimator read 1.5e-2 drift on an **exact** solution until the Dirichlet node was
included, and 6.5e-5 after; convergence level 4 measured its own roundoff floor
and scored a perfect solver at 0.48; the stochastic correlation-time estimator
carries a 10-26% bias on exact data. Each is recorded where the constant lives.

**A.6 — `params` is the leak nobody would notice.**
The review asked whether an agent could read the answer from the manifest or the
parameters before executing. `TaskInstance` is what an adapter serialises, so
`params` is public — and nothing in it looks like an answer, which is exactly why
it needs a check rather than a convention.
`tests/test_no_truth_travels.py` recomputes each task's truth independently and
asserts it appears in no prompt and no `params`, across all five families at
once, so a sixth inherits it.

**A.7 — The sandbox is not a security boundary, and says so.**
Review suggested Docker/gVisor/WASM for v2. Agreed and deferred; what is not
deferred is the honesty. `core/sandbox.py` states that a determined escape from
in-process Python is possible and that the real boundary is a container — which
the hub runtimes already provide. A "sandboxed" claim a reviewer breaks in five
minutes costs more credibility than the feature buys.

**A.8 — A schema error must be actionable, or the RL signal goes sparse.**
Adopted. Every refusal returns `breakdown.reason` and, where relevant,
`schema_error: True`, and the refinement demo feeds exactly that text back. The
reward rose because the breakdown said something a model could act on; a verifier
that only ever answered "wrong" could not drive the loop.
