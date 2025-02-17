import os

def buscar_archivo_por_nombre(nombre_archivo, unidad="M:/"):
    rutas_encontradas = []
    for raiz, _, archivos in os.walk(unidad):
        if nombre_archivo in archivos:
            rutas_encontradas.append(os.path.join(raiz, nombre_archivo))
    
    return rutas_encontradas

nombre_a_buscar = "intents.json"
rutas = buscar_archivo_por_nombre(nombre_a_buscar)

if rutas:
    for ruta in rutas:
        print("Archivo encontrado en:", ruta)
else:
    print("Archivo no encontrado")
