"""
Módulo de Tools para Agentes IA
Contiene todas las herramientas disponibles para los agentes.
"""

from tools.Base_de_conocimiento import buscar_datapath
from tools.Busqueda_internet import buscar_internet

# Lista de todas las tools disponibles
__all__ = [
    "buscar_datapath",
    "buscar_internet",
]
