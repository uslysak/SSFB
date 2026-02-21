
from paramater import parameter_parser
from load_comm import G
from seed_select import select_seed_nodes
from SeGNE import SeGNE
from First_Stage import seed_based_communities
from metric import cal_EQ


def main():
    # 1) 读参数
    args = parameter_parser()

    # 2) 选 seed
    seeds_node = select_seed_nodes(G, num_clusters=args.cluster_number)
    print("Raw seeds:", seeds_node)
    args.seed_nodes = seeds_node

    # 3) 训练 SeGNE，得到 embedding
    model = SeGNE(args, G)
    emb = model.train()

    return emb, seeds_node


if __name__ == "__main__":
    emb ,seeds_node = main()
    first_communities, assignments = seed_based_communities(
        G,
        emb,
        seeds_node,
        threshold=0.5,
        normalize=True
    )
    # 输出示例
    print("社区数量:", len(first_communities))
    for seed, comm in first_communities.items():
        print(f"Seed {seed}: 社区大小 = {len(comm)}")
        print(f"  节点: {sorted(comm)}")  # 排序后更好看
    print(f"初始社区EQ:{cal_EQ(first_communities,G)}")


