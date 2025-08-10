import numpy as np
from skimage import io, morphology
from skimage.morphology import skeletonize
from scipy.ndimage import convolve
from scipy.spatial.distance import pdist
import networkx as nx
import argparse


def calcular_metricas(path_imagen):
    # Cargar imagen binaria
    imagen = io.imread(path_imagen, as_gray=True)
    binaria = imagen > 0.5

    # Esqueleto
    esqueleto = skeletonize(binaria)

    # Contar vecinos
    vecinos = convolve(esqueleto.astype(np.uint8), np.ones((3, 3)), mode='constant') - esqueleto

    bifurcaciones = (esqueleto & (vecinos > 2))
    extremos = (esqueleto & (vecinos == 1))

    num_bifurcaciones = np.count_nonzero(bifurcaciones)
    num_extremos = np.count_nonzero(extremos)
    longitud_total = np.count_nonzero(esqueleto)

    # Crear grafo
    coords = np.column_stack(np.nonzero(esqueleto))
    G = nx.Graph()
    for y, x in coords:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                ny, nx_ = y + dy, x + dx
                if 0 <= ny < esqueleto.shape[0] and 0 <= nx_ < esqueleto.shape[1]:
                    if esqueleto[ny, nx_]:
                        G.add_edge((y, x), (ny, nx_))

    # Ramas
    ramas = []
    visitados = set()
    for nodo in G.nodes:
        if nodo in visitados:
            continue
        if G.degree[nodo] == 1:
            camino = [nodo]
            visitados.add(nodo)
            actual = nodo
            previo = None
            while True:
                vecinos = [n for n in G.neighbors(actual) if n != previo]
                if not vecinos:
                    break
                previo = actual
                actual = vecinos[0]
                camino.append(actual)
                visitados.add(actual)
                if G.degree[actual] != 2:
                    break
            if len(camino) > 1:
                ramas.append(camino)

    longitudes_ramas = [len(rama) for rama in ramas]
    media_rama = np.mean(longitudes_ramas)

    coord_bif = np.column_stack(np.nonzero(bifurcaciones))
    if len(coord_bif) > 1:
        dist_bif = pdist(coord_bif)
        media_dist_bif = np.mean(dist_bif)
        std_dist_bif = np.std(dist_bif)
    else:
        media_dist_bif = 0
        std_dist_bif = 0

    # Resultados
    resultados = {
        "Numero de bifurcaciones": num_bifurcaciones,
        "Numero de extremos": num_extremos,
        "Longitud total del esqueleto": longitud_total,
        "Longitud media de ramas": media_rama,
        "Distancia media entre bifurcaciones": media_dist_bif,
        "Desviacion estandar de distancias entre bifurcaciones": std_dist_bif
    }
    return resultados


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculo de metricas de venacion a partir de una mascara binaria")
    parser.add_argument("imagen", help="Ruta a la imagen binaria de entrada")
    args = parser.parse_args()

    metricas = calcular_metricas(args.imagen)
    for k, v in metricas.items():
        print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")
