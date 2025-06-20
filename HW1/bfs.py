import csv
edgeFile = 'edges.csv'


def bfs(start, end):
    # Begin your code (Part 1)
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
    q = []
    vis = set()
    q.append((start, [start], 0.0))
    vis.add(start)
    while len(q):
        now, path, now_dist = q.pop(0)
        if now == end:
            return path, now_dist, len(vis)
        for u, d in ed[now]:
            if u not in vis:
                vis.add(u)
                nw_path = path + [u]
                nw_dist = now_dist + d
                q.append((u, nw_path, nw_dist))

    # raise NotImplementedError("To be implemented")
    # End your code (Part 1)


if __name__ == '__main__':
    path, dist, num_visited = bfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
