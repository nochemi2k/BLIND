"""
Common ligands and counterions of coordination chemistry, for turning formulas such as
[Cu(phen)2Cl]ClO4 or names such as dichlorobis(triphenylphosphine)palladium into SMILES.

Each entry is (SMILES, charge). Complexes are written as disconnected ions (metal cation +
ligands with their usual charge), so the element composition and the total charge are exact,
while the metal-ligand bonds are not drawn.
"""

# Abbreviations and formula-like ligands used inside [ ... ] in formulas
LIGANDS = {
    # halides and pseudohalides
    "F": ("[F-]", -1), "Cl": ("[Cl-]", -1), "Br": ("[Br-]", -1), "I": ("[I-]", -1),
    "CN": ("[C-]#N", -1), "NCS": ("[N-]=C=S", -1), "SCN": ("[S-]C#N", -1), "NCO": ("[N-]=C=O", -1),
    "N3": ("[N-]=[N+]=[N-]", -1), "NCSe": ("[N-]=C=[Se]", -1),
    "H": ("[H-]", -1), "OH": ("[OH-]", -1), "O": ("[O-2]", -2), "S": ("[S-2]", -2), "N": ("[N-3]", -3),
    "OMe": ("C[O-]", -1), "OEt": ("CC[O-]", -1), "OiPr": ("CC(C)[O-]", -1), "OtBu": ("CC(C)(C)[O-]", -1),
    "OPh": ("[O-]c1ccccc1", -1), "SPh": ("[S-]c1ccccc1", -1),
    # oxyanions
    "NO3": ("[O-][N+](=O)[O-]", -1), "NO2": ("N(=O)[O-]", -1), "ONO2": ("[O-][N+](=O)[O-]", -1),
    "ClO4": ("[O-]Cl(=O)(=O)=O", -1), "SO4": ("[O-]S(=O)(=O)[O-]", -2), "CO3": ("[O-]C([O-])=O", -2),
    "C2O4": ("[O-]C(=O)C([O-])=O", -2), "ox": ("[O-]C(=O)C([O-])=O", -2), "PO4": ("[O-]P([O-])([O-])=O", -3),
    "OAc": ("CC(=O)[O-]", -1), "O2CMe": ("CC(=O)[O-]", -1), "O2CCH3": ("CC(=O)[O-]", -1), "OOCCH3": ("CC(=O)[O-]", -1),
    "CH3COO": ("CC(=O)[O-]", -1), "AcO": ("CC(=O)[O-]", -1), "OBz": ("[O-]C(=O)c1ccccc1", -1),
    "O2CCF3": ("[O-]C(=O)C(F)(F)F", -1), "TFA": ("[O-]C(=O)C(F)(F)F", -1), "OPiv": ("CC(C)(C)C(=O)[O-]", -1),
    "HCOO": ("[O-]C=O", -1), "OTf": ("[O-]S(=O)(=O)C(F)(F)F", -1), "CF3SO3": ("[O-]S(=O)(=O)C(F)(F)F", -1),
    "OTs": ("Cc1ccc(S(=O)(=O)[O-])cc1", -1), "NTf2": ("[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F", -1),
    # weakly coordinating anions
    "PF6": ("F[P-](F)(F)(F)(F)F", -1), "BF4": ("F[B-](F)(F)F", -1), "SbF6": ("F[Sb-](F)(F)(F)(F)F", -1),
    "AsF6": ("F[As-](F)(F)(F)(F)F", -1), "BPh4": ("c1ccc([B-](c2ccccc2)(c2ccccc2)c2ccccc2)cc1", -1),
    "BArF": ("FC(F)(F)c1cc([B-](c2cc(C(F)(F)F)cc(C(F)(F)F)c2)(c2cc(C(F)(F)F)cc(C(F)(F)F)c2)c2cc(C(F)(F)F)cc(C(F)(F)F)c2)cc(C(F)(F)F)c1", -1),
    # neutral small ligands
    "CO": ("[C-]#[O+]", 0), "NO": ("[N]=O", 0), "H2O": ("O", 0), "OH2": ("O", 0), "aq": ("O", 0),
    "NH3": ("N", 0), "MeCN": ("CC#N", 0), "CH3CN": ("CC#N", 0), "NCMe": ("CC#N", 0), "NCCH3": ("CC#N", 0),
    "PhCN": ("N#Cc1ccccc1", 0), "NCPh": ("N#Cc1ccccc1", 0), "CS": ("[C-]#[S+]", 0),
    "MeOH": ("CO", 0), "CH3OH": ("CO", 0), "EtOH": ("CCO", 0), "thf": ("C1CCOC1", 0), "THF": ("C1CCOC1", 0),
    "dmso": ("CS(C)=O", 0), "DMSO": ("CS(C)=O", 0), "dmf": ("CN(C)C=O", 0), "DMF": ("CN(C)C=O", 0),
    "Et2O": ("CCOCC", 0), "OEt2": ("CCOCC", 0), "dme": ("COCCOC", 0), "DME": ("COCCOC", 0),
    "py": ("c1ccncc1", 0), "Py": ("c1ccncc1", 0), "pyr": ("c1ccncc1", 0), "DMAP": ("CN(C)c1ccncc1", 0),
    "en": ("NCCN", 0), "tmeda": ("CN(C)CCN(C)C", 0), "TMEDA": ("CN(C)CCN(C)C", 0),
    "dien": ("NCCNCCN", 0), "trien": ("NCCNCCNCCN", 0), "tren": ("NCCN(CCN)CCN", 0),
    "cyclam": ("C1CNCCNCCCNCCNC1", 0), "tacn": ("C1CNCCNCCN1", 0),
    "bpy": ("c1ccc(-c2ccccn2)nc1", 0), "bipy": ("c1ccc(-c2ccccn2)nc1", 0), "2,2'-bpy": ("c1ccc(-c2ccccn2)nc1", 0),
    "4,4'-bpy": ("c1cc(-c2ccncc2)ccn1", 0), "dmbpy": ("Cc1ccnc(-c2cc(C)ccn2)c1", 0),
    "dtbbpy": ("CC(C)(C)c1ccnc(-c2cc(C(C)(C)C)ccn2)c1", 0), "phen": ("c1cnc2c(c1)ccc1cccnc12", 0),
    "Phen": ("c1cnc2c(c1)ccc1cccnc12", 0), "dmp": ("Cc1ccc2ccc3ccc(C)nc3c2n1", 0),
    "neocuproine": ("Cc1ccc2ccc3ccc(C)nc3c2n1", 0), "bphen": ("c1ccc(-c2ccnc3c2ccc2c(-c4ccccc4)ccnc23)cc1", 0),
    "tpy": ("c1ccc(-c2cccc(-c3ccccn3)n2)nc1", 0), "terpy": ("c1ccc(-c2cccc(-c3ccccn3)n2)nc1", 0),
    "dppz": ("c1ccc2nc3c(nc2c1)c1cccnc1c1ncccc31", 0), "bpm": ("c1cnc(-c2ncccn2)nc1", 0),
    "ppy": ("[c-]1ccccc1-c1ccccn1", -1), "piq": ("[c-]1ccccc1-c1nccc2ccccc12", -1),
    "PPh3": ("c1ccc(P(c2ccccc2)c2ccccc2)cc1", 0), "PMe3": ("CP(C)C", 0), "PEt3": ("CCP(CC)CC", 0),
    "PCy3": ("C1CCC(P(C2CCCCC2)C2CCCCC2)CC1", 0), "PtBu3": ("CC(C)(C)P(C(C)(C)C)C(C)(C)C", 0),
    "P(OPh)3": ("c1ccc(OP(Oc2ccccc2)Oc2ccccc2)cc1", 0), "P(OMe)3": ("COP(OC)OC", 0), "PMe2Ph": ("CP(C)c1ccccc1", 0),
    "PMePh2": ("CP(c1ccccc1)c1ccccc1", 0), "AsPh3": ("c1ccc([As](c2ccccc2)c2ccccc2)cc1", 0),
    "dppm": ("c1ccc(P(CP(c2ccccc2)c2ccccc2)c2ccccc2)cc1", 0), "dppe": ("c1ccc(P(CCP(c2ccccc2)c2ccccc2)c2ccccc2)cc1", 0),
    "dppp": ("c1ccc(P(CCCP(c2ccccc2)c2ccccc2)c2ccccc2)cc1", 0), "dppb": ("c1ccc(P(CCCCP(c2ccccc2)c2ccccc2)c2ccccc2)cc1", 0),
    "dppf": ("[CH-]1C=CC=C1P(c1ccccc1)c1ccccc1.[CH-]1C=CC=C1P(c1ccccc1)c1ccccc1.[Fe+2]", 0),
    "dcpe": ("C1CCC(P(CCP(C2CCCCC2)C2CCCCC2)C2CCCCC2)CC1", 0), "dmpe": ("CP(C)CCP(C)C", 0),
    "xantphos": ("CC1(C)c2cccc(P(c3ccccc3)c3ccccc3)c2Oc2c(P(c3ccccc3)c3ccccc3)cccc21", 0),
    "Xantphos": ("CC1(C)c2cccc(P(c3ccccc3)c3ccccc3)c2Oc2c(P(c3ccccc3)c3ccccc3)cccc21", 0),
    "BINAP": ("c1ccc(P(c2ccccc2)c2ccc3ccccc3c2-c2c(P(c3ccccc3)c3ccccc3)ccc3ccccc23)cc1", 0),
    "OPPh3": ("O=P(c1ccccc1)(c1ccccc1)c1ccccc1", 0), "SMe2": ("CSC", 0), "tht": ("C1CCSC1", 0),
    # pi ligands and chelating anions
    "Cp": ("[cH-]1cccc1", -1), "C5H5": ("[cH-]1cccc1", -1), "Cp*": ("C[c-]1c(C)c(C)c(C)c1C", -1),
    "C5Me5": ("C[c-]1c(C)c(C)c(C)c1C", -1), "Ind": ("[cH-]1ccc2ccccc12", -1),
    "cod": ("C1=CCCC=CCC1", 0), "COD": ("C1=CCCC=CCC1", 0), "nbd": ("C1=CC2C=CC1C2", 0), "NBD": ("C1=CC2C=CC1C2", 0),
    "cot": ("C1=CC=CC=CC=C1", 0), "C2H4": ("C=C", 0), "C6H6": ("c1ccccc1", 0), "p-cymene": ("Cc1ccc(C(C)C)cc1", 0),
    "cymene": ("Cc1ccc(C(C)C)cc1", 0), "p-cym": ("Cc1ccc(C(C)C)cc1", 0), "pcym": ("Cc1ccc(C(C)C)cc1", 0), "C6Me6": ("Cc1c(C)c(C)c(C)c(C)c1C", 0), "allyl": ("[CH2-]C=C", -1),
    "C3H5": ("[CH2-]C=C", -1), "dba": ("O=C(C=Cc1ccccc1)C=Cc1ccccc1", 0),
    "acac": ("CC(=O)[CH-]C(C)=O", -1), "hfac": ("O=C([CH-]C(=O)C(F)(F)F)C(F)(F)F", -1),
    "tfac": ("CC(=O)[CH-]C(=O)C(F)(F)F", -1), "dbm": ("O=C([CH-]C(=O)c1ccccc1)c1ccccc1", -1),
    "thd": ("CC(C)(C)C(=O)[CH-]C(=O)C(C)(C)C", -1), "tmhd": ("CC(C)(C)C(=O)[CH-]C(=O)C(C)(C)C", -1),
    "dpm": ("CC(C)(C)C(=O)[CH-]C(=O)C(C)(C)C", -1), "TTA": ("O=C([CH-]C(=O)C(F)(F)F)c1cccs1", -1),
    "tta": ("O=C([CH-]C(=O)C(F)(F)F)c1cccs1", -1), "pic": ("[O-]C(=O)c1ccccn1", -1), "hq": ("[O-]c1cccc2cccnc12", -1),
    "q": ("[O-]c1cccc2cccnc12", -1), "salen": ("[O-]c1ccccc1C=NCCN=Cc1ccccc1[O-]", -2),
    "gly": ("NCC(=O)[O-]", -1), "edta": ("[O-]C(=O)CN(CCN(CC(=O)[O-])CC(=O)[O-])CC(=O)[O-]", -4),
    "dtc": ("[S-]C(=S)N", -1), "Et2dtc": ("CCN(CC)C(=S)[S-]", -1), "S2CNEt2": ("CCN(CC)C(=S)[S-]", -1),
    "Tp": ("[BH-](n1cccn1)(n1cccn1)n1cccn1", -1),
    # alkyl and aryl
    "Me": ("[CH3-]", -1), "CH3": ("[CH3-]", -1), "Et": ("C[CH2-]", -1), "Ph": ("[c-]1ccccc1", -1), "C6H5": ("[c-]1ccccc1", -1),
    "C6F5": ("F[c-]1c(F)c(F)c(F)c1F", -1), "Mes": ("Cc1cc(C)[c-]c(C)c1", -1), "Bn": ("[CH2-]c1ccccc1", -1),
    "CH2SiMe3": ("C[Si](C)(C)[CH2-]", -1), "C≡CPh": ("[C-]#Cc1ccccc1", -1),
    # counter cations
    "NBu4": ("CCCC[N+](CCCC)(CCCC)CCCC", 1), "Bu4N": ("CCCC[N+](CCCC)(CCCC)CCCC", 1), "NEt4": ("CC[N+](CC)(CC)CC", 1),
    "Et4N": ("CC[N+](CC)(CC)CC", 1), "NMe4": ("C[N+](C)(C)C", 1), "PPh4": ("c1ccc([P+](c2ccccc2)(c2ccccc2)c2ccccc2)cc1", 1),
    "Ph4P": ("c1ccc([P+](c2ccccc2)(c2ccccc2)c2ccccc2)cc1", 1), "PPN": ("c1ccc(P(=[N+]=P(c2ccccc2)(c2ccccc2)c2ccccc2)(c2ccccc2)c2ccccc2)cc1", 1),
    "NH4": ("[NH4+]", 1), "HNEt3": ("CC[NH+](CC)CC", 1), "Li": ("[Li+]", 1), "Na": ("[Na+]", 1), "K": ("[K+]", 1),
    "Cs": ("[Cs+]", 1), "Rb": ("[Rb+]", 1),
}

# Ligand names in additive nomenclature (dichlorobis(triphenylphosphine)palladium)
LIGAND_NAMES = {
    "fluoro": "F", "fluorido": "F", "chloro": "Cl", "chlorido": "Cl", "bromo": "Br", "bromido": "Br",
    "iodo": "I", "iodido": "I", "cyano": "CN", "cyanido": "CN", "hydrido": "H", "hydroxo": "OH", "hydroxido": "OH",
    "oxo": "O", "oxido": "O", "azido": "N3", "thiocyanato": "SCN", "isothiocyanato": "NCS", "nitrato": "NO3",
    "nitrito": "NO2", "acetato": "OAc", "trifluoroacetato": "TFA", "triflato": "OTf", "perchlorato": "ClO4",
    "methoxo": "OMe", "methoxido": "OMe", "ethoxo": "OEt", "ethoxido": "OEt", "isopropoxo": "OiPr", "tert-butoxo": "OtBu",
    "carbonyl": "CO", "nitrosyl": "NO", "aqua": "H2O", "ammine": "NH3", "acetonitrile": "MeCN", "pyridine": "py",
    "methyl": "Me", "ethyl": "Et", "phenyl": "Ph", "benzyl": "Bn", "pentafluorophenyl": "C6F5", "mesityl": "Mes",
    "cyclopentadienyl": "Cp", "pentamethylcyclopentadienyl": "Cp*", "η5-cyclopentadienyl": "Cp",
    "η5-pentamethylcyclopentadienyl": "Cp*", "1,5-cyclooctadiene": "cod", "cycloocta-1,5-diene": "cod",
    "η4-1,5-cyclooctadiene": "cod", "norbornadiene": "nbd", "allyl": "allyl", "η3-allyl": "allyl",
    "p-cymene": "p-cymene", "η6-p-cymene": "p-cymene", "benzene": "C6H6", "η6-benzene": "C6H6",
    "acetylacetonato": "acac", "pentane-2,4-dionato": "acac", "hexafluoroacetylacetonato": "hfac",
    "triphenylphosphine": "PPh3", "triphenylphosphane": "PPh3", "trimethylphosphine": "PMe3", "triethylphosphine": "PEt3",
    "tricyclohexylphosphine": "PCy3", "tri-tert-butylphosphine": "PtBu3", "triphenylarsine": "AsPh3",
    "triphenylphosphine oxide": "OPPh3", "dimethyl sulfoxide": "dmso", "dimethylsulfoxide": "dmso",
    "tetrahydrofuran": "thf", "ethylenediamine": "en", "ethane-1,2-diamine": "en",
    "2,2'-bipyridine": "bpy", "2,2′-bipyridine": "bpy", "2,2'-bipyridyl": "bpy", "2,2′-bipyridyl": "bpy",
    "1,10-phenanthroline": "phen", "2,2':6',2''-terpyridine": "tpy", "2,2′:6′,2″-terpyridine": "tpy",
    "1,2-bis(diphenylphosphino)ethane": "dppe", "bis(diphenylphosphino)methane": "dppm",
    "1,1'-bis(diphenylphosphino)ferrocene": "dppf", "1,1′-bis(diphenylphosphino)ferrocene": "dppf",
    "2-phenylpyridinato": "ppy", "2-phenylpyridine": "ppy", "8-quinolinolato": "q", "quinolin-8-olato": "q",
    "picolinato": "pic", "oxalato": "ox", "glycinato": "gly", "dimethylformamide": "dmf", "N,N-dimethylformamide": "dmf",
}

METAL_NAMES = {
    "lithium": "Li", "sodium": "Na", "potassium": "K", "magnesium": "Mg", "calcium": "Ca", "strontium": "Sr", "barium": "Ba",
    "scandium": "Sc", "titanium": "Ti", "vanadium": "V", "chromium": "Cr", "manganese": "Mn", "iron": "Fe", "cobalt": "Co",
    "nickel": "Ni", "copper": "Cu", "zinc": "Zn", "yttrium": "Y", "zirconium": "Zr", "niobium": "Nb", "molybdenum": "Mo",
    "ruthenium": "Ru", "rhodium": "Rh", "palladium": "Pd", "silver": "Ag", "cadmium": "Cd", "hafnium": "Hf",
    "tantalum": "Ta", "tungsten": "W", "rhenium": "Re", "osmium": "Os", "iridium": "Ir", "platinum": "Pt", "gold": "Au",
    "mercury": "Hg", "aluminium": "Al", "aluminum": "Al", "gallium": "Ga", "indium": "In", "thallium": "Tl", "tin": "Sn",
    "lead": "Pb", "bismuth": "Bi", "lanthanum": "La", "cerium": "Ce", "praseodymium": "Pr", "neodymium": "Nd",
    "samarium": "Sm", "europium": "Eu", "gadolinium": "Gd", "terbium": "Tb", "dysprosium": "Dy", "holmium": "Ho",
    "erbium": "Er", "thulium": "Tm", "ytterbium": "Yb", "lutetium": "Lu", "thorium": "Th", "uranium": "U",
}
METALS = set(METAL_NAMES.values()) | {"Be", "Tc", "Fr", "Ra", "Ac", "Pa", "Np", "Pu", "Am"}
