rule download_genomes:
    output:
        "data/genomes/genome_list.txt"

    shell:
        """
        python scripts/00_download_genomes.py \
            --max_genomes {config[genome_limit]}
        """
