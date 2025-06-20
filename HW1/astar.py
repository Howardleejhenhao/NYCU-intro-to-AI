import csv
import heapq

edgeFile = 'edges.csv'
heuristicFile = 'heuristic_values.csv'


def astar(start, end):
    # Begin your code (Part 4)
    ed = {}
    with open(edgeFile, 'r', encoding='utf-8-sig') as f:
        all = csv.DictReader(f)
        for now in all:
            u = int(now['start'])
            v = int(now['end'])
            d = float(now['distance'])
            if u not in ed: ed[u] = []
            if v not in ed: ed[v] = []
            ed[u].append((v, d))
            # ed[v].append((u, d))
    h = {}
    with open(heuristicFile, 'r', encoding='utf-8-sig') as f:
        all = csv.DictReader(f)
        for row in all:
            node_id = int(row['node'])
            h[node_id] = {
                1079387396: float(row['1079387396']),
                1737223506: float(row['1737223506']),
                8513026827: float(row['8513026827'])
            }

    cost = {start: 0.0}
    pq = []
    start_cst = cost[start] + h[start][end]
    heapq.heappush(pq, (start_cst, start, [start], 0.0))
    vis = set()
    while len(pq):
        fh, now, path, f = heapq.heappop(pq)
        if now == end:
            return path, f, len(vis)
        
        if now in vis:
            continue
        vis.add(now)
        
        for neighbor, dist in ed[now]:
            new_g = f + dist
            if new_g < cost.get(neighbor, float('inf')):
                cost[neighbor] = new_g
                neighbor_h = h.get(neighbor, {}).get(end, 0.0)
                new_f = new_g + neighbor_h
                new_path = path + [neighbor]
                heapq.heappush(pq, (new_f, neighbor, new_path, new_g))
            
    raise NotImplementedError("To be implemented")
    # End your code (Part 4)


if __name__ == '__main__':
    path, dist, num_vis = astar(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of vis nodes: {num_vis}')
