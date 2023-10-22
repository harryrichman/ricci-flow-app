""" sage code, from Mark Kempton """

def Omega(G): #Resistance matrix of G
    return G.effective_resistance_matrix(nonedgesonly=False)

def p(G): #vector of node resistances
    L = G.laplacian_matrix()
    O=Omega(G)
    I = matrix.identity(len(G.vertices()))
    return vector((I+.5*L*O)[:,0])

def pTOmegap(G): #p^T*Omega*p
    O = Omega(G); 
    L = G.laplacian_matrix()
    return 0.5 * (O * L * O)[0,0]

def k(G): #vector of node resistances using the new paper definition
    return (1 / pTOmegap(G)) * p(G)

def show_with_resistances(G):
    O=Omega(G); G.relabel()
    for e in G.edges():
        G.set_edge_label(e[0],e[1],round(1.0*O[e[0],e[1]],3))
    G.show(edge_labels=True)
