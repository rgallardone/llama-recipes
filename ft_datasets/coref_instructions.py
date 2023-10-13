INSTRUCTIONS = {
    "full_text": {
        "en": {
            "inst1": """Find all the entities that corefer on the following text. You should enclose
             the corefering entities with square brackets and add a number that identifies each
             entity, like this: '[1 John] is a farmer in [2 Arizona]. [1 He] is the best on
             [2 the state]'."""
        },
        "es": {
            "inst1": """Encuentra todas las entidades que correfieren en el siguiente texto.
             Encierra cada entidad con parentesis rectos y añade un número que identifique cada
             entidad, como en el siguiente ejemplo: '[1 Juan] es un granjero en [2 Uruguay].
             [1 Él] es el mejor de [2 el país].""",
            "inst2": """Identifica que entidades correfieren en el siguiente texto. Las entidades
             fueron encerradas con parentesis rectos, y se debe añadir un número que identifique
             cada entidad dentro de los parentesis rectos, como en el siguiente ejemplo: '[1 Juan]
             es un granjero en [2 Uruguay]. [1 Él] es el mejor de [2 el país].""",
            "inst3": """Identifica que menciones correfieren en el texto. Las menciones fueron
             encerradas con parentesis rectos, y se debe añadir un número dentro de los parentesis
             rectos que identifique la entidad a la que refieren las menciones. Notar que hay
             entidades que pueden estar dentro de otras; en ese caso, se debe detectar el
             identificador para cada entidad anidada. Por ejemplo, para el texto de entrada
             '[ Juan ] es un granjero en [ Uruguay ]. [ Él ] es el mejor de [ el país ].',
             la salida deberia ser '[1 Juan] es un granjero en [2 Uruguay]. [1 Él] es el mejor de
             [2 el país].'. El texto está delimitado por los tokens <texto> y </texto>.""",
        },
    },
    "next_sentence": {
        "es": {
            "inst1": (
                "Dado un texto encerrado por los tokens <texto> y </texto> y una oracion "
                "encerrada por los tokens <oracion> y </oracion> con menciones a entidades, "
                "identificar en la oración a que entidad refiere cada mención. En el texto y la "
                "oracion se identifica con parentesis rectos las menciones a entidades. A su vez, "
                "en el texto se identifica con un número todas las menciones que refieren a una "
                "misma entidad. Se debe agregar un identificador a cada mención en la oración que "
                "la asocie a una entidad en el texto. Si no refiere a una entidad presente en el "
                "texto, agrega un nuevo identificador. Responde unicamente con la oracion resultante "
                "y nada mas."
            ),
            "inst1_first": (
                "Dada una oracion encerrada por los tokens <oracion> y </oracion> con "
                "menciones a entidades, identificar con un número a que entidad refiere cada mención. "
                "En la oración se identifica con parentesis rectos las menciones a entidades. "
                "Las menciones a la misma entidad deben llevar el mismo identificador, y todas las "
                "entidades distintas deben llevar distintos identificadores. Responde unicamente con "
                "la oracion resultante y nada mas."
            ),
        }
    },
}

INSTRUCTIONS_MENTION = {
    "inst": (
        "Dado un texto encerrado por los tokens <texto> y </texto> con menciones a entidades y "
        "el indice de la palabra cabeza, y la siguiente oración del texto encerrada por los "
        "tokens <oracion> y </oracion>, identificar en la oración las menciones a entidades y el "
        "indice de su palabra cabeza. Las menciones se identifican con parentesis rectos, y el "
        "indice de la palabra cabeza se identifica con 'head<i>' donde <i> es el indice dentro de "
        "la mención de la palabra cabeza. Responde unicamente con la oración resultante y nada más."
    ),
    "inst_first": (
        "Dada una oración encerrada por los tokens <oracion> y </oracion>, identificar las "
        "menciones a entidades y el indice de la palabra cabeza de la mención. Las menciones se "
        "identifican con parentesis rectos, y el indice de la palabra cabeza se identifica con "
        "'head<i>' donde <i> es el indice dentro de la mención de la palabra cabeza. Responde "
        "unicamente con la oración resultante y nada más."
    ),
}
