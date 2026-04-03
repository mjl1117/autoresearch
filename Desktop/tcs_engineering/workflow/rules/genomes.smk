rule download_genome:
    output:
        proteins="data/genomes/{genome}/proteins.faa",
        genome="data/genomes/{genome}/genome.fna",
        gff="data/genomes/{genome}/genome.gff"
    params:
        # Use our new universal dictionary
        ftp=lambda wc: FTP_LOOKUP[wc.genome]
    shell:
        """
        PREFIX=$(basename {params.ftp})
        curl -L {params.ftp}/${{PREFIX}}_protein.faa.gz | gunzip > {output.proteins}
        curl -L {params.ftp}/${{PREFIX}}_genomic.fna.gz | gunzip > {output.genome}
        curl -L {params.ftp}/${{PREFIX}}_genomic.gff.gz | gunzip > {output.gff}
        """
