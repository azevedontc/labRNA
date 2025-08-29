import turtle
import numpy as np

from lab.busca import sorteia_de_lista, sorteia_coords
from lab.busca.agente import Agente
from lab.busca.alvo import Alvo
from lab.busca.grade import Grade

rnd = np.random.default_rng(1650)
grade = Grade()
agente = Agente(grade, linha=-19, coluna=-19)
alvo = Alvo(grade, *sorteia_coords(grade, rnd))

while agente != alvo:
    direcao = sorteia_de_lista(["norte", "sul", "leste", "oeste"], rnd)
    agente.move(direcao)

turtle.done()
