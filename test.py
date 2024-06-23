L_samples = ["A4-1","A4-2", "A4-3", "A5-1", "A5-2", "A5-3", "A6-1", "A6-2", "A6-3", "A6-4"]

matrix = []
for sample in L_samples:
    csv_dir = './spots'
    csv_path = f'{sample}-cellspots.csv'

    df = pd.read_csv(os.path.join(csv_dir, csv_path))
    kl=[]
    points = df[["dim_1", "dim_2"]].values
    X=points[:,1]
    Y=points[:,0]
    x = np.linspace(X.min(), X.max(), int((X.max()-X.min())/10))
    y = np.linspace(Y.min(), Y.max(), int((Y.max()-Y.min())/10))
    X_grid, Y_grid = np.meshgrid(x, y)
    xy = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T
    bandwidth = 2
    
    for i in tqdm(range(len(unique_genes))):
        points1 = df[df["gene"]==unique_genes[i]][["dim_1", "dim_2"]].values
        kde1 = KernelDensity(bandwidth=bandwidth).fit(points1)
        density1 = np.exp(kde1.score_samples(xy))
        kl.append(density1)
    # similarity=np.zeros((len(unique_genes), len(unique_genes)))
    # cos=np.zeros((len(unique_genes), len(unique_genes)))
    w_dis=np.zeros((len(unique_genes), len(unique_genes)))
    for i in tqdm(range(len(unique_genes))):
        for n in range(len(unique_genes)):
            # simi = -np.log(np.sum(np.minimum(kl[i], kl[n])) / np.sum(np.maximum(kl[i], kl[n])))
            # similarity[i][n]=simi
            # cosine_similarity = 1 - distance.cosine(kl[i], kl[n])
            # cos[i][n]=cosine_similarity
            w_dist = wasserstein_distance(range(xy.shape[0]),range(xy.shape[0]),kl[i], kl[n])
            w_dis[i][n]=w_dist

    virus_index = [unique_genes.index(gene) for gene in virus_genes]
    host_index = [unique_genes.index(gene) for gene in host_genes]
    sub_matrix = w_dis[virus_index][:, host_index]
    matrix.append(sub_matrix)