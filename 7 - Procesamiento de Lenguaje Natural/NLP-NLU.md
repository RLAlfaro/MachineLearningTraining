##

1. NLP: [N]atural [L]anguage [P]rocessing
2. NLU: [N]atural [L]anguage [U]nderstanding

## CORPUS - Coleccion de muchos textos

**CORPORA:** Coleccion de colecciones de Texto

### Normalizacion de Texto

1. Tokenizacion: Separar en palabras toda la cadena de texto
2. Lematizacion: Convertir cada palabra en su raiz fundamental
3. Segmentacion: Separar las frases, con comas y puntos 

## PMI

Comentario a Procesar:

P(w1) = frec(w1)/Total_tokens
P(w2) = frec(w2)/Total_tokens
P(w1, w2) = frec(w1, w2)/Total_bigramas

entonces:

PMI = P(w1, w2) / [P(w1) P(w2)]
PMI = (Total_tokens**2/Total_bigramas) * frec(w1, w2) / [frec(w1)frec(w2)]

PMI = constante * frec(w1, w2) / [frec(w1)frec(w2)]

Esto quiere decir que calcular esta metrica por probabilidades o por conteos es equivalente salvo un factor constante que es el mismo para todos los bigramas dentro del mismo corpus. Como son equivalentes, decido usar la metrica calculada por conteos porque es mas sencillo 😃

En cuanto a tu segunda pregunta, es común ver en algunos libros que decidan usar el blog en base 2, pero no hay problema si quieres usar otra base para el logaritmo, es un tema de convención, lo importante es que siempre uses la misma base de los para comparar diferentes cadenas de texto.


