from edges import get_distance_centroid

def get_shape_sig(edge_dict):
    shapesig_dict = {}
    for i in range(len(edge_dict.items())):
        edges = list(edge_dict.values())[i] # edges at time point i
        shape_signatures = [get_distance_centroid(edges[j]) for j in range(len(edges))]

        time = list(edge_dict.keys())[i]

        shapesig_dict[time] = shape_signatures
    return shapesig_dict
