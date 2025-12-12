# ACPCCodes

# Persistent Segment Tree 
#(Standard)
```cpp
struct Vertex {
    Vertex *l, *r;
    int sum;

    Vertex(int val) : l(nullptr), r(nullptr), sum(val) {}
    Vertex(Vertex *l, Vertex *r) : l(l), r(r), sum(0) {
        if (l) sum += l->sum;
        if (r) sum += r->sum;
    }
};

Vertex* build(int a[], int tl, int tr) {
    if (tl == tr)
        return new Vertex(a[tl]);
    int tm = (tl + tr) / 2;
    return new Vertex(build(a, tl, tm), build(a, tm+1, tr));
}

int get_sum(Vertex* v, int tl, int tr, int l, int r) {
    if (l > r)
        return 0;
    if (l == tl && tr == r)
        return v->sum;
    int tm = (tl + tr) / 2;
    return get_sum(v->l, tl, tm, l, min(r, tm))
         + get_sum(v->r, tm+1, tr, max(l, tm+1), r);
}

Vertex* update(Vertex* v, int tl, int tr, int pos, int new_val) {
    if (tl == tr)
        return new Vertex(new_val);
    int tm = (tl + tr) / 2;
    if (pos <= tm)
        return new Vertex(update(v->l, tl, tm, pos, new_val), v->r);
    else
        return new Vertex(v->l, update(v->r, tm+1, tr, pos, new_val));
}
```

# find k-th smallest number in a range
```cpp
Vertex* build(int tl, int tr) {
    if (tl == tr)
        return new Vertex(0);
    int tm = (tl + tr) / 2;
    return new Vertex(build(tl, tm), build(tm+1, tr));
}

Vertex* update(Vertex* v, int tl, int tr, int pos) {
    if (tl == tr)
        return new Vertex(v->sum+1);
    int tm = (tl + tr) / 2;
    if (pos <= tm)
        return new Vertex(update(v->l, tl, tm, pos), v->r);
    else
        return new Vertex(v->l, update(v->r, tm+1, tr, pos));
}

int find_kth(Vertex* vl, Vertex *vr, int tl, int tr, int k) {
    if (tl == tr)
        return tl;
    int tm = (tl + tr) / 2, left_count = vr->l->sum - vl->l->sum;
    if (left_count >= k)
        return find_kth(vl->l, vr->l, tl, tm, k);
    return find_kth(vl->r, vr->r, tm+1, tr, k-left_count);
}
```

# Rerooting
```cpp
#include <bits/stdc++.h>
using namespace std;

vector<int> graph[200001];
int fir[200001], sec[200001], ans[200001];

void dfs1(int node = 1, int parent = 0) {
	for (int i : graph[node])
		if (i != parent) {
			dfs1(i, node);
			if (fir[i] + 1 > fir[node]) {
				sec[node] = fir[node];
				fir[node] = fir[i] + 1;
			} else if (fir[i] + 1 > sec[node]) {
				sec[node] = fir[i] + 1;
			}
		}
}

void dfs2(int node = 1, int parent = 0, int to_p = 0) {
	ans[node] = max(to_p, fir[node]);
	for (int i: graph[node])
		if (i != parent) {
			if (fir[i] + 1 == fir[node]) dfs2(i, node, max(to_p, sec[node]) + 1);
			else dfs2(i, node, ans[node] + 1);
		}
}

int main() {
	ios_base::sync_with_stdio(0);
	cin.tie(0);
	int n;
	cin >> n;
	for (int i = 1; i < n; i++) {
		int u, v;
		cin >> u >> v;
		graph[u].push_back(v);
		graph[v].push_back(u);
	}
	dfs1();
	dfs2();
	for (int i = 1; i <= n; i++) cout << ans[i] << ' ';
	return 0;
}
```

# Centriod
```cpp
#include <cstring>
#include <iostream>
#include <vector>
#define ll long long
using namespace std;
const int N=2e5+5;
int n,m,k,q;
vector<int> g[N];
int is_removed[N]={},sz[N];
int cnt[N],mx_sz=0;
ll ans=0;
int get_sz(int v,int p) {
    sz[v]=1;
    for (auto u:g[v]) {
        if (u==p || is_removed[u]) continue;
        sz[v]+=get_sz(u,v);
    }
    return sz[v];
}
int get_centroid(int v,int p,int cur_sz) {
    for (auto u:g[v]) {
        if (u==p || is_removed[u])continue;
        if (sz[u]*2>cur_sz) return get_centroid(u,v,cur_sz);
    }
    return v;
}
void dfs_add(int v,int p,int depth,int delta) {
    cnt[depth]+=delta;
    mx_sz=max(mx_sz,depth);
    for (auto u:g[v]) {
        if (u==p || is_removed[u])continue;
        dfs_add(u,v,depth+1,delta);
    }
}
void calc(int v,int p,int depth) {
    if (depth<=k)
        ans+=cnt[k-depth];
    for (auto u:g[v]) {
        if (u==p || is_removed[u])continue;
        calc(u,v,depth+1);
    }
}
void decompose(int v) {
    int cur_sz=get_sz(v,0);
    int cent=get_centroid(v,0,cur_sz);
    // ans
    cnt[0]=1;
    mx_sz=0;
    for (auto u:g[cent]) {
        if (is_removed[u]) continue;
        calc(u,cent,1);
        dfs_add(u,cent,1,1);
    }
    //
    for (int i=0;i<=mx_sz;++i) {
        cnt[i]=0;
    }
    is_removed[cent]=1;
    for (auto u:g[cent]) {
        if (is_removed[u]) continue;
        decompose(u);
    }
}
int main() {
    ios_base::sync_with_stdio(false),cin.tie(0),cout.tie(0);
    cin>>n>>k;
    int u,v;
    for (int i=1;i<n;i++) {
        cin>>u>>v;
        g[u].push_back(v);
        g[v].push_back(u);
    }
    decompose(1);
    cout<<ans<<'\n';
    return 0;
}
```
#centriod 2 
```cpp
#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>
#define ll long long
using namespace std;
const int N=2e5+5;
int n,m,k,q;
vector<int> g[N];
int is_removed[N]={},sz[N];
int cnt[N],tot_cnt[N],pref[N],mx_depth=0,sub_depth;
int depth[N];
ll ans=0;
int k1,k2;
int get_sz(int v,int p) {
    sz[v]=1;
    depth[v]=1;
    for (auto u:g[v]) {
        if (u==p || is_removed[u]) continue;
        sz[v]+=get_sz(u,v);
        depth[v]+=depth[u];
    }
    return sz[v];
}
int get_centroid(int v,int p,int cur_sz) {
    for (auto u:g[v]) {
        if (u==p || is_removed[u])continue;
        if (sz[u]*2>cur_sz) return get_centroid(u,v,cur_sz);
    }
    return v;
}
void dfs_add(int v,int p,int depth,int delta) {
    cnt[depth]+=delta;
    sub_depth=max(sub_depth,depth);
    for (auto u:g[v]) {
        if (u==p || is_removed[u])continue;
        dfs_add(u,v,depth+1,delta);
    }
}
 
void decompose(int v) {
    int cur_sz=get_sz(v,0);
    int cent=get_centroid(v,0,cur_sz);
    // ans
    mx_depth=0;
    tot_cnt[0]=1;
    ll intial=(k1==1);
    for (auto u:g[cent]) {
        if (is_removed[u])continue;
        sub_depth=0;
        dfs_add(u,cent,1,1);
        ll sum=intial;
        for (int i=1;i<=sub_depth;++i) {
            ans+=sum*cnt[i];
            if (k2-i>=0)sum-=tot_cnt[k2-i];
            if (k1-i-1>=0)sum+=tot_cnt[k1-i-1];
        }
        for (int i=k1-1;i<k2&& i<=sub_depth;++i) {
            intial+=cnt[i];
        }
        for (int i=0;i<=sub_depth;++i) {
            tot_cnt[i]+=cnt[i];
            cnt[i]=0;
        }
        mx_depth=max(mx_depth,sub_depth);
    }
    //
    for (int i=0;i<=mx_depth;++i) {
        tot_cnt[i]=0;
    }
    is_removed[cent]=1;
    for (auto u:g[cent]) {
        if (is_removed[u]) continue;
        decompose(u);
    }
}
int main() {
    ios_base::sync_with_stdio(false),cin.tie(0),cout.tie(0);
    cin>>n>>k1>>k2;
    int u,v;
    for (int i=1;i<n;i++) {
        cin>>u>>v;
        g[u].push_back(v);
        g[v].push_back(u);
    }
    decompose(1);
    cout<<ans<<'\n';
    return 0;
}

```

#centroid 3
```cpp
#include <bits/stdc++.h>
#define pp pair<int,int>
#define ll long long
using namespace std;

#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace __gnu_pbds;

struct Pair {
    ll val;
    int id;
    bool operator<(const Pair& other) const {
        if (val != other.val) return val < other.val;
        return id < other.id;
    }
};

using ordered_multiset = tree<
    Pair,
    null_type,
    less<>,
    rb_tree_tag,
    tree_order_statistics_node_update>;

const int N=2e5+5;
int n,m,k,q;

const int LOG = 20;
vector<pp> g[N];
int up[N][LOG];
ll depth[N],val[N],len[N];
int is_removed[N]={},sz[N];
int parent[N]={};
int get_sz(int v,int p) {
    sz[v]=1;
    for (auto [u,w]:g[v]) {
        if (u==p || is_removed[u])continue;
        sz[v]+=get_sz(u,v);
    }
    return sz[v];
}
int get_centroid(int v,int p,int cur_sz) {
    for (auto [u,w]:g[v]) {
        if (u==p || is_removed[u])continue;
        if (sz[u]*2>cur_sz)
            return get_centroid(u,v,cur_sz);
    }
    return v;
}

ordered_multiset os;
vector<pair<ll,int>> qr[N];
ll ans[N];
int qid=0;
void calc(int v,int p,ll depth) {
    for (auto [x,id]:qr[v]) {
        if (x>=depth){
            ans[id]+=os.order_of_key({x-depth+1,-1});
            //ans[id]+=get(1,0,1000,0,x-depth);
        }
    }
    for (auto [u,w]:g[v]) {
        if (u==p || is_removed[u]) continue;
        calc(u,v,depth+w);
    }
}
void add_dfs(int v,int p,ll depth,int delta) {
    // update(1,0,1000,depth,delta);
    if (delta==1)
        os.insert({depth,qid++});
    else os.erase(os.lower_bound({depth,-1}));
    for (auto [u,w]:g[v]) {
        if (u==p || is_removed[u]) continue;
        add_dfs(u,v,depth+w,delta);
    }
}

int decompose(int v) {
    int cur_sz=get_sz(v,0);
    int cent=get_centroid(v,0,cur_sz);
    // calc
    // update(1,0,1000,0,1);
    os.insert({0,qid++});
    for (auto [u,w]:g[cent]) {
        if (is_removed[u]) continue;
        add_dfs(u,cent,w,1);
    }
    for (auto [x,id]:qr[cent]) {
        ans[id]+=os.order_of_key({x+1,-1});
    }
    for (auto [u,w]:g[cent]) {
        if (is_removed[u]) continue;
        add_dfs(u,cent,w,-1);
        calc(u,cent,w);
        add_dfs(u,cent,w,1);
    }

    // clear
    os.clear();
    is_removed[cent]=1;
    for (auto [u,w]:g[cent]) {
        if (is_removed[u])continue;
        int child=decompose(u);
        parent[child]=cent;
    }
    return cent;
}

int main() {
    ios_base::sync_with_stdio(false),cin.tie(0),cout.tie(0);
    cin>>n>>m;
    int u,v,w;
    for (int i=1;i<n;i++) {
        cin>>u>>v>>w;
        g[u].push_back({v,w});
        g[v].push_back({u,w});
    }
    ll t=0,x,c,d,time=0;
    for (int i=0;i<m;i++) {
        cin>>v>>d;
        qr[v].push_back({d,t++});
    }
    parent[decompose(1)]=-1;
    for (int i=0;i<m;i++) {
        cout<<ans[i]<<'\n';
    }
    return 0;
}

```

#Max Flow
```cpp
#include <bits/stdc++.h>
#define pp pair<int,int>
#define ll long long
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace __gnu_pbds;
using namespace std;
template <typename T>
using ordered_set = tree<T,null_type,less<T>,rb_tree_tag,tree_order_statistics_node_update>;
const ll N=500+5,INF=1e18;
ll n,m;
ll capacity[N][N];
vector<int> adj[N];
 
ll bfs(int s, int t, vector<int>& parent) {
    fill(parent.begin(), parent.end(), -1);
    parent[s] = -2;
    queue<pair<ll, ll>> q;
    q.push({s, INF});
 
    while (!q.empty()) {
        ll cur = q.front().first;
        ll flow = q.front().second;
        q.pop();
 
        for (int next : adj[cur]) {
            if (parent[next] == -1 && capacity[cur][next]) {
                parent[next] = cur;
                ll new_flow = min(flow, capacity[cur][next]);
                if (next == t)
                    return new_flow;
                q.push({next, new_flow});
            }
        }
    }
 
    return 0;
}
 
ll maxflow(int s, int t) {
    ll flow = 0;
    vector<int> parent(n);
    ll new_flow;
 
    while (new_flow = bfs(s, t, parent)) {
        flow += new_flow;
        int cur = t;
        while (cur != s) {
            int prev = parent[cur];
            capacity[prev][cur] -= new_flow;
            capacity[cur][prev] += new_flow;
            cur = prev;
        }
    }
 
    return flow;
}
int main() {
    ios_base::sync_with_stdio(false),cin.tie(0),cout.tie(0);
    cin>>n>>m;
    ll u,v,c;
    for (int i=1;i<=m;i++) {
        cin>>u>>v>>c;
        u--;v--;
        adj[u].push_back(v);
        adj[v].push_back(u);
        capacity[u][v] += c;
    }
    cout<<maxflow(0,n-1);
    cout<<'\n';
    return 0;
 
}
```

# Max flow min cut
```cpp
#include <bits/stdc++.h>
#define pp pair<int,int>
#define ll long long
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace __gnu_pbds;
using namespace std;
template <typename T>
using ordered_set = tree<T,null_type,less<T>,rb_tree_tag,tree_order_statistics_node_update>;
const ll N=500+5,INF=1e18;
ll n,m;
int capacity[N][N];
vector<int> adj[N];
 
int bfs(int s, int t, vector<int>& parent) {
    fill(parent.begin(), parent.end(), -1);
    parent[s] = -2;
    queue<pair<int, int>> q;
    q.push({s, INF});
 
    while (!q.empty()) {
        int cur = q.front().first;
        int flow = q.front().second;
        q.pop();
 
        for (int next : adj[cur]) {
            if (parent[next] == -1 && capacity[cur][next]) {
                parent[next] = cur;
                int new_flow = min(flow, capacity[cur][next]);
                if (next == t)
                    return new_flow;
                q.push({next, new_flow});
            }
        }
    }
 
    return 0;
}
 
int maxflow(int s, int t) {
    int flow = 0;
    vector<int> parent(n);
    int new_flow;
 
    while (new_flow = bfs(s, t, parent)) {
        flow += new_flow;
        int cur = t;
        while (cur != s) {
            int prev = parent[cur];
            capacity[prev][cur] -= new_flow;
            capacity[cur][prev] += new_flow;
            cur = prev;
        }
    }
 
    return flow;
}
int vis[N];
void dfs(int s,int t) {
    vis[s]=t;
    for (auto v : adj[s]) {
        if (!vis[v]) {
            if (capacity[s][v] || t==2)
                dfs(v,t);
        }else if (vis[v]==1 && t==2) {
            cout<<s+1<<' '<<v+1<<'\n';
        }
    }
}
int main() {
    ios_base::sync_with_stdio(false),cin.tie(0),cout.tie(0);
    cin>>n>>m;
    ll u,v;
    for (int i=1;i<=m;i++) {
        cin>>u>>v;
        u--;v--;
        adj[u].push_back(v);
        adj[v].push_back(u);
        capacity[u][v] += 1;
        capacity[v][u] += 1;
    }
    cout<<maxflow(0,n-1)<<'\n';
    dfs(0,1);
    dfs(n-1,2);
    return 0;
 
}
```

# 2D Segment tree
```cpp
#include<bits/stdc++.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include<vector>
#include<map>
#include<list>
#include<stack>
#include<queue>
#include<deque>
#include<set>
#include<string.h>
#define ll long long
using namespace std;
const int N=1e5+5;
const ll MOD=1e9+7,MAX=1e18;
const double ep=1e-6, pi=3.14159265359;
int dx[8] = { 1, 0, -1, 0, -1, 1, -1, 1 };
int dy[8] = { 0, 1, 0, -1, -1, 1, 1, -1 };
ll n,m=0,k;
void fastio()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
}
using namespace std;
ll a[1001][1001];    
ll arr[4000][4000];
        
struct segtree{
    int size;
    void init(int n){
        size=1;
        while(size<n)size*=2;
        // arr.resize(size,vector<ll>(size,0));
        // for(int i=0;i<size*2;i++)arr.push_back(NEUel);
    }
    void build_y(int x,int lx,int rx,int y,int ly,int ry)
    {
        if (ry-ly == 1) {
        if (lx == rx-1)
            arr[x][y] = a[lx][ly];
        else
            arr[x][y] = arr[x*2+1][y] + arr[x*2+2][y];
    } else {
        int my = (ly + ry) / 2;
        build_y(x, lx, rx, y*2+1, ly, my);
        build_y(x, lx, rx, y*2+2, my, ry);
        arr[x][y] = arr[x][y*2+1] + arr[x][y*2+2];
    }
    }
    void build_x(int x,int lx,int rx)
    {
        if(rx-lx!=1){
            int m=(lx+rx)/2;
            build_x(2*x+1,lx,m);
            build_x(2*x+2,m,rx);
        }
        build_y(x,lx,rx,0,0,size);
    //arr[x]=merge(arr[2*x+1],arr[2*x+2]);
    }
    void build()
    {
        build_x(0,0,size);
    }
    void update_y(int x, int lx, int rx, int y, int ly, int ry, int a, int b, int new_val) {
        if (ly == ry-1) {
            if (lx == rx-1)
            {
                arr[x][y] = (arr[x][y]+1)%2;
                // cout<<arr[x][y]<<'\n';
            }
            else
                arr[x][y] = arr[x*2+1][y] + arr[x*2+2][y];
        } else {
            int my = (ly + ry) / 2;
            if (b < my)
                update_y(x, lx, rx, y*2+1, ly, my, a,b, new_val);
            else
                update_y(x, lx, rx, y*2+2, my, ry, a, b, new_val);
            arr[x][y] = arr[x][y*2+1] + arr[x][y*2+2];
        }
    }

    void update_x(int x, int lx, int rx, int a, int b, int new_val) {
    if (lx != rx-1) {
        int mx = (lx + rx) / 2;
        if (a < mx)
            update_x(x*2+1, lx, mx, a, b, new_val);
        else
            update_x(x*2+2, mx, rx, a, b, new_val);
    }
    update_y(x, lx, rx, 0, 0, size, a, b, new_val);
    }
    void update_x(int a,int b,int val=0){
        update_x(0,0,size,a,b,val);
    }

    int sum_y(int x, int y, int tly, int try_, int ly, int ry) {
        if (tly>=ry||try_<=ly) 
            return 0;
        if (ly <= tly && try_ <= ry)
            return arr[x][y];
        int tmy = (tly + try_) / 2;
        return sum_y(x, y*2+1, tly, tmy, ly, ry)
            + sum_y(x, y*2+2, tmy, try_, ly, ry);
    }
    int sum_x(int x, int tlx, int trx, int lx, int rx, int ly, int ry) {
        if (tlx>=rx||trx<=lx)
            return 0;
        if (lx <= tlx && trx <= rx)
            return sum_y(x, 0, 0,size, ly, ry);
        int tmx = (tlx + trx) / 2;
        return sum_x(x*2+1, tlx, tmx, lx, rx, ly, ry)
            + sum_x(x*2+2, tmx, trx, lx, rx, ly, ry);
    }
    int sum(int a,int b,int c,int d)
    {
        return sum_x(0,0,size,a,b,c,d);
    }
};    

void fn()
{
    cin>>n>>m;
    segtree st;
    st.init(n);
    string s;
    for(int i=0;i<n;i++)
    {
        cin>>s;
        for(int j=0;j<n;j++)
        {
            a[i][j]=(int)(s[j]=='*');
            //cout<<a[i][j]<<" ";
        }
        cout<<'\n';
    }
    // cout<<1;
    st.build();
    // return;
    while(m--)
    {
        int c;
        ll a,b;
        cin>>c>>a>>b;
        if(c==2)
        {
            ll d,f;
            cin>>d>>f;
            cout<<st.sum(a-1,d,b-1,f)<<'\n';
        }
        else
        {
            st.update_x(a-1,b-1);
        }
    }
}
int main()
{
    // fastio();
    int t=1;
    // cin>>t;
    while(t--)fn();
    return 0;
}
```

# 2D Spare Table
```cpp
int f[N][N][10][10];
    int query(int l, int d, int r, int u)
    {
        int k1 = std::__lg(r - l);
        int k2 = std::__lg(u - d);
        return std::max({f[l][d][k1][k2], f[r - (1 << k1)][d][k1][k2], f[l][u - (1 << k2)][k1][k2], f[r - (1 << k1)][u - (1 << k2)][k1][k2]});
    }
    // build
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            f[i][j][0][0] = a[i][j];
        }
    }

for (int k1 = 0; k1 < 10; k1++)
{
    for (int k2 = 0; k2 < 10; k2++)
    {
        for (int i = 0; i + (1 << k1) <= n; i++)
        {
            for (int j = 0; j + (1 << k2) <= m; j++)
            {
                if (k1 == 0 && j + (2 << k2) <= m)
                {
                    f[i][j][k1][k2 + 1] = std::max(f[i][j][k1][k2], f[i][j + (1 << k2)][k1][k2]);
                }
                if (i + (2 << k1) <= n)
                {
                    f[i][j][k1 + 1][k2] = std::max(f[i][j][k1][k2], f[i + (1 << k1)][j][k1][k2]);
                }
            }
        }
    }
}
```
