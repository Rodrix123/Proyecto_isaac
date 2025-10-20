import random as rd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from joblib import Parallel, delayed
from concurrent.futures import ProcessPoolExecutor, as_completed

def generate_matrix(seed = None ):

    
    if seed is None:
        seed = rd.randint(0,10000000)
            
    rd.seed(seed)
    np.random.seed(seed)
    print(f"Seed: {seed}")
    matrix = np.zeros((13, 13), dtype=int)
    initial_pos = np.random.randint(0, 13, size = (1,2))
    matrix[initial_pos[0,0],initial_pos[0,1]] = 1
    print(f"Initial Position: {initial_pos}")
    num_rooms = rd.randint(7, 15)
    
    print(f"Number of Rooms: {num_rooms}")
    num_rooms -= 1
    rooms = [initial_pos]

    while num_rooms > 0:
        new_rooms = []
        for room in rooms:
            temp,num_rooms = _generate_matrix(room, matrix,num_rooms)
            new_rooms.extend(temp)

        rooms = new_rooms        
    return matrix

def _generate_matrix(pos,matrix,num_rooms):
        adjacent  = np.array([[-1,0],[1,0],[0,-1],[0,1]])
        adjacent_rooms= [pos + adjacent[i] for i in range(4) if 0<= pos[0,0] + adjacent[i,0] < 13 and 0<= pos[0,1] + adjacent[i,1] < 13 and matrix[pos[0,0] + adjacent[i,0], pos[0,1] + adjacent[i,1]] == 0]
        selected = min(rd.randint(1,4),num_rooms,len(adjacent_rooms))
        selected_adjacent = np.random.choice(len(adjacent_rooms),selected,replace=False)
        adjacent_rooms= [adjacent_rooms[i] for i in selected_adjacent]
        for j in adjacent_rooms:
            matrix[j[0,0],j[0,1]] = 1
            num_rooms -= 1
        return adjacent_rooms,num_rooms
""""def generate_matrix_BFS(seed = None):
    
    
    if seed is None:
        seed = rd.randint(0,10000000) 
    rd.seed(seed)
    np.random.seed(seed)
    print(f"Seed: {seed}")
    matrix = np.zeros((13, 13), dtype=int)
    initial_pos = np.random.randint(0, 13, size = (2,1))
    matrix[initial_pos[0],initial_pos[0]] = 1
    print(f"Initial Position: {initial_pos}")
    num_rooms = rd.randint(7, 15)
    
    print(f"Number of Rooms: {num_rooms}")
    num_rooms -= 1
    queue = [initial_pos]
    pos_frontier  = np.array([[-1,0],[1,0],[0,-1],[0,1]]).T
    min_rooms = 1
    while num_rooms > 0:
        print(f"Rooms left: {num_rooms}, Queue length: {len(queue)}")
        print(f"Queue: {queue}")
        pos = queue.pop(0)

        adjacent  =(np.array([pos for _ in range(4)]).T+ pos_frontier)
        print(f"adjacent: {adjacent}")
        valid_adjacent = [j for j in adjacent if j[0,0] >= 0 and j[0,0] < 13 and j[0,1]>= 0 and j[0,1] < 13 and matrix[j[0,0],j[0,1]] == 0]
        if len(valid_adjacent) > 0:
            selected = min(rd.randint(min_rooms,len(valid_adjacent)),num_rooms)
            selected_adjacent = np.random.choice(len(valid_adjacent),selected,replace=False)
            adjacent_rooms= [valid_adjacent[i] for i in selected_adjacent]
            for j in adjacent_rooms:
                matrix[j[0,0],j[0,1]] = 1
                queue.append(j)
                num_rooms -= 1
        min_rooms = 0   


    return matrix"""
def generate_matrix_points(seed = None):
     
    
    if seed is None:
        seed = rd.randint(0,10000000)
    seed =1    
    rd.seed(seed)
    np.random.seed(seed)
    print(f"Seed: {seed}")
    matrix = np.zeros((13, 13), dtype=int)
    initial_pos = np.random.randint(0, 13, size = (1,2))
    matrix[initial_pos[0,0],initial_pos[0,1]] = 1
    print(f"Initial Position: {initial_pos}")
    num_rooms = rd.randint(7, 15)
    special_rooms_pos = np.random.randint(0, 13, size = (rd.randint(3,6),2))
    print(f"Special Rooms Position: {special_rooms_pos}")
    print(f"Number of Rooms: {num_rooms}")
    num_rooms -= 1
    queue = [initial_pos]


    return matrix         
def show_matrix(matrixa):
    size_x, size_y = matrixa.shape
    fig, ax = plt.subplots()

    # mostrar la matriz (0 = blanco, 1 = negro)
    ax.imshow(matrixa, cmatrix="gray_r", origin="upper")

    # dibujar líneas de la cuadrícula
    ax.set_xticks(np.arange(-0.5, size_y, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, size_x, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=1)

    # quitar labels de ejes
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()
#print(generate_matrix_points())
"""matrixa =  generate_matrix(1)
show_matrix(matrixa)
print(matrixa)"""

def generate_map_Graph(seed = None):
    
    
    if seed is None:
        seed = rd.randint(0,10000000)
            
    rd.seed(seed)
    np.random.seed(seed)
    print(f"Seed: {seed}")
    G = nx.Graph()
    initial_pos = (rd.randint(3,9),rd.randint(3,9))
    print(f"Initial pos {initial_pos}")
    G.add_node(initial_pos)
    
    num_rooms = rd.randint(7, 15)
    

    print(f"Number of Rooms: {num_rooms}")
    num_rooms -= 1
    queue = [initial_pos]
    adjacent =  [[-1,0],[1,0],[0,-1],[0,1]]  

    while num_rooms > 0:
        actual_room = queue.pop(0)
        valid_adjacent = [(actual_room[0] + adj[0], actual_room[1] + adj[1]) for adj in adjacent if 0<= actual_room[0] + adj[0] <=12 and 0<= actual_room[1] + adj[1] <=12 and (actual_room[0] + adj[0], actual_room[1] + adj[1]) not in G.nodes]
        if len(valid_adjacent) > 0:
            selected = min(rd.randint(1,len(valid_adjacent)),num_rooms)
            selected_adjacent = rd.sample(valid_adjacent,selected)
            for adj in selected_adjacent:
                G.add_node(adj)
                G.add_edge(actual_room,adj)
                for neigh_adj in [(adj[0] + adj_[0], adj[1] + adj_[1]) for adj_ in adjacent if(adj[0] + adj_[0], adj[1] + adj_[1]) in G.nodes] :
                    G.add_edge(adj,neigh_adj)
                queue.append(adj)
                num_rooms -= 1 
    return G   
def Terminal_rooms(graph, n):
    Full = True
    adjacent =  [[-1,0],[1,0],[0,-1],[0,1]] 
    t_rooms =  [node for node in graph.nodes if graph.degree(node) == 1] 
    n_t = len(t_rooms)
    print(f"Current terminal rooms: {n_t}")
    possible_t_rooms = []
    if n_t < n:
        Full = False
        not_t_rooms = graph.nodes - t_rooms
        for room in not_t_rooms:
            valid_adjacent = [(room[0] + adj[0], room[1] + adj[1]) for adj in adjacent if 0<= room[0] + adj[0] <=12 and 0<= room[1] + adj[1] <=12 and (room[0] + adj[0], room[1] + adj[1]) not in graph.nodes]
            for adj in valid_adjacent:
                if [(adj_[0] + adj[0], adj_[1] + adj[1]) for adj_ in adjacent if (adj[0] + adj_[0], adj[1] + adj_[1]) in graph.nodes] == [room]:
                    graph.add_node(adj)
                    graph.add_edge(room,adj)
                    t_rooms.append(adj)
                    n_t += 1
                    if n_t == n:
                        break
            if n_t == n:
                break                

            
 
        

    return t_rooms  ,Full  
def More_terminal_rooms(graph, n):
        adjacent =  [[-1,0],[1,0],[0,-1],[0,1]] 
        t_rooms =  [node for node in graph.nodes if graph.degree(node) == 1] 
        n_t = len(t_rooms)
        t_rooms = rd.shuffle(t_rooms)
        for room in t_rooms:
            if n_t == n:
                break
            valid_adjacent = [(room[0] + adj[0], room[1] + adj[1]) for adj in adjacent if 0<= room[0] + adj[0] <=12 and 0<= room[1] + adj[1] <=12 and (room[0] + adj[0], room[1] + adj[1]) not in graph.nodes]
            
            for adj in valid_adjacent:
                Valid_adj = False
                graph.add_node(adj)
                graph.add_edge(room,adj)
                t_rooms.remove(room)
                valid_adjacent_new = [(adj_[0] + adj[0], adj_[1] + adj[1]) for adj_ in adjacent if 0<= adj[0] + adj_[0] <=12 and 0<= adj[1] + adj_[1] <=12 and (adj[0] + adj_[0], adj[1] + adj_[1]) not in graph.nodes]
                for new_terminal in valid_adjacent_new:
                    if [(new_terminal[0] + adj_[0], new_terminal[1] + adj_[1]) for adj_ in adjacent if (new_terminal[0] + adj_[0], new_terminal[1] + adj_[1]) in graph.nodes] == [adj]:
                        graph.add_node(new_terminal)
                        graph.add_edge(adj,new_terminal)
                      
                        n_t += 1
                        Valid_adj = True
                        if n_t == n:
                            break

                if not Valid_adj:
                    n_t 
                    graph.remove_node(adj)

                
def Plot_map(graph):
    # Diccionario {nodo: (x,y)} usando las mismas coordenadas de los nodos
    pos = {node: (node[0], node[1]) for node in graph.nodes}
    
    plt.figure(figsize=(6,6))
    nx.draw(
        graph, pos,
        with_labels=True,       # muestra etiquetas (las coordenadas)
        node_size=500,
        node_color="skyblue",
        font_size=8,
        font_color="black",
        edgecolors="black"
    )
    plt.gca().set_aspect('equal', adjustable='box')  # mantener cuadrícula
    plt.show()

def map_islands(graph: nx.Graph): #unpolished
    rooms = list(graph.nodes)
    rooms.sort()
    islands = []
    visited = set()
    adjacent = [[(1,0), (0,1), (1,1)],[(-1,0), (0,-1), (-1,-1)],[(-1,0), (0,1), (-1,1)],[(1,0), (0,-1), (1,-1)]]
    for room in rooms:
        if room in visited:
            continue
        for adj in adjacent:
            existing_island = False
            adj_= ((room[0] + adj[0][0], room[1] + adj[0][1]), (room[0] + adj[1][0], room[1] + adj[1][1]), (room[0] + adj[2][0], room[1] + adj[2][1]))
            if all(a in graph.nodes for a in adj_):
                for island in islands:    
                    if adj_[0] in island or adj_[1] in island or adj_[2] in island:
                        island.update({room, adj_[0], adj_[1], adj_[2]})
                        existing_island = True
                    
                if not existing_island:    
                    islands.append({room, adj_[0], adj_[1], adj_[2]})
                
                for i in adj_:
                    visited.add(i)
    return islands           
            
graph= generate_map_Graph(123489)
Plot_map(graph)
a,b =Terminal_rooms(graph,6)

print(map_islands(graph))
Plot_map(graph)