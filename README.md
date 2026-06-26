# InstrinsicMotivation-ARC-AGI

Un agente de **aprendizaje por refuerzo** que aprende a resolver un rompecabezas
tipo *ARC-AGI* mirando solo los píxeles de la pantalla, usando **motivación
intrínseca** (curiosidad) para explorar cuando casi no hay recompensa.

## El problema

El juego se llama **Bloom Chain**. Una semilla hace crecer flores que avanzan por
la tierra; hay que guiarlas para que toquen las flores de colores **en el orden
correcto, una por una**. Si dos llegan al mismo tiempo, o en el orden equivocado,
la cadena se rompe.

Lo difícil para una IA es que la **recompensa es escasa**: el agente solo recibe
una señal (+1) cuando *gana*. El resto del tiempo no recibe ninguna pista. En un
juego así, un agente normal casi nunca llega a ganar por casualidad, así que nunca
aprende nada. Ese es el reto central del proyecto.

## La idea

Para que el agente aprenda sin recompensas constantes, combinamos tres cosas:

- **Visión por píxeles + PPO.** Una red neuronal convolucional (CNN) lee la
  pantalla y un algoritmo estándar de RL (PPO) decide las acciones.
- **Curiosidad (motivación intrínseca).** El agente se premia a sí mismo por
  visitar situaciones nuevas o por *aprender* algo, lo que lo empuja a explorar.
  Comparamos dos métodos: **RND** (novedad) y **LPM** (progreso de aprendizaje,
  2025), más robusto al "ruido".
- **Currículo inverso.** En lugar de empezar con el tablero más difícil,
  empezamos por situaciones fáciles donde ganar está al alcance, y subimos la
  dificultad poco a poco solo cuando el agente ya domina el nivel actual.

## El resultado

El agente aprendió a resolver el primer nivel **incluso con el tablero totalmente
desordenado**, ganando alrededor del **80 %** de las partidas.

La conclusión honesta del proyecto es interesante: la curiosidad **ayuda**, pero
no fue lo más importante. Lo que de verdad hizo posible el aprendizaje fueron el
**currículo** (hacer que ganar fuera alcanzable) y el **rediseño de las acciones**
(convertir un intercambio de fichas en dos "clics" en lugar de una larga
secuencia de movimientos). En otras palabras: *la motivación intrínseca es
necesaria, pero no suficiente.*

## Cómo está organizado

- `bloom_env.py`, `bloom_click_env.py` — el juego como entorno de entrenamiento.
- `agent.py` — las redes (PPO + curiosidad RND/LPM).
- `click_experiment.py` — entrena, compara métodos y genera gráficas.
- `play_live.py` — ver al agente jugar en tiempo real.
- `bloom_chain.html` — el juego para que lo juegue una persona.

## Jugar el juego

¿Quieres probarlo tú mismo? Es el **mismo juego y los mismos controles** que usó
la IA. Haz clic en la semilla para crecer y haz clic en dos casillas de tierra
para intercambiarlas.

**▶ [Jugar Bloom Chain](https://il-palazzos.itch.io/bloomchain)**

Contraseña: `iguala`

<!-- Reemplaza PON-AQUI-TU-ENLACE con la URL de tu juego (itch.io o Vercel). -->
