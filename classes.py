from pydantic import BaseModel

class Sintomas(BaseModel):
    TOSSE: bool = False
    DORES_MUSCULARES: bool = False
    CANSACO: bool = False
    DOR_DE_GARGANTA: bool = False
    CORIZA: bool = False
    NARIZ_ENTUPIDO: bool = False
    FEBRE: bool = False
    NAUSEA: bool = False
    VOMITOS: bool = False
    DIARREIA: bool = False
    FALTA_DE_AR: bool = False
    DIFICULDADE_PARA_RESPIRAR: bool = False
    PERDA_DE_PALADAR: bool = False
    PERDA_DE_OLFATO: bool = False
    COCEIRA_NO_NARIZ: bool = False
    OLHOS_COM_COCEIRA: bool = False
    COCEIRA_NA_BOCA: bool = False
    COCEIRA_NO_OUVIDO_INTERNO: bool = False
    ESPIRROS: bool = False
    CONJUNTIVITE: bool = False
