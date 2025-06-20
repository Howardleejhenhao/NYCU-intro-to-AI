import csv
import heapq
edgeFile = 'edges.csv'


def ucs(start, end):
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
    pq = [(0.0, start, [start])]
    vis = set()
    cost = {start: 0.0}

    while len(pq):
        now_cost, now, path = heapq.heappop(pq)
        if now == end:
            return path, now_cost, len(vis)
        if now in vis:
            continue
        vis.add(now)
        for u, d in ed[now]:
            nw_cost = now_cost + d
            if u not in vis or nw_cost < cost[u]:
                cost[u] = nw_cost
                heapq.heappush(pq, (nw_cost, u, path + [u]))

    # raise NotImplementedError("To be implemented")
    # End your code (Part 3)


if __name__ == '__main__':
    path, dist, num_visited = ucs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
