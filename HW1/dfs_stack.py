import csv
edgeFile = 'edges.csv'

def dfs(start, end):
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
    vis = set()
    stack = [(start, [start], 0.0)]

    while stack:
        now, path, now_dist = stack.pop()
        if now not in vis:
            vis.add(now)
            if now == end: return path, now_dist, len(vis)
            for u, d in ed[now]:
                if u not in vis:
                    stack.append((u, path + [u], now_dist + d))


    # raise NotImplementedError("To be implemented")
    # End your code (Part 2)


if __name__ == '__main__':
    path, dist, num_visited = dfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
