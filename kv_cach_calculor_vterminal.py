
class Modele:
    def __init__(self, nom:str, editeur:str, archi_type:str, ):
        self.nom= nom
        self.editeur= editeur
        self.archi= archi_type

        self.nbr_attention_heads_kv: int =None
        self.nbr_head_dim: int =None
        self.nbr_Gated_Attention_layers: int =None
        self.quantization_oct: int =None


    def input_model_parameters(self, nbr_attention_heads_kv: int, nbr_head_dim: int,
                        nbr_Gated_Attention_layers: int,  quantization_oct:int ):
        """
        - nbr_attention_heads_kv: Number of Attention Heads for KV. 
        - nbr_Gated_Attention_layers: nombre de couches Gated Attention, déduit du Hidden Layout. Attention pas de DeltaNet qui n ont pas de KV cache!
        - quantization_oct: 4=fp32, 2=bf16/fp16, 1=fp8/int8
        """
        if quantization_oct not in (1,2,4):
            raise ValueError("quantization_oct doit etre 1,2 ou 4")
        
        self.nbr_attention_heads_kv=nbr_attention_heads_kv
        self.nbr_head_dim=nbr_head_dim
        self.nbr_Gated_Attention_layers=nbr_Gated_Attention_layers
        self.quantization_oct=quantization_oct
        return "done"

    def calculer_kv_cache(self,  length_seq : int):
        """
        - length_seq: longueur de séquence 
        """

        # step 1 : calculer éléments par token et par couche
        elmt_per_token_per_layer=2*self.nbr_attention_heads_kv*self.nbr_head_dim
        
        # step 2: multiplié par tokens et couches
        # Chaque bloc contient 1 couche d'attention, le bloc est répété N fois (nbr_Gated_Attention_layers)
        elements=elmt_per_token_per_layer*length_seq*(self.nbr_Gated_Attention_layers)

        # step 3 : Des éléments aux octets
        kv_cache_oct=elements*self.quantization_oct
        kv_cache=kv_cache_oct/(2**20)

        return kv_cache

def main():
    print("=== Calculateur de KV Cache ===")
    print("a rtx-llm-engine projet ")

    nom = str(input("Nom du modele : "))
    editeur = str(input("Editeur du modele : "))
    archi_type = str(input("ARchitecture : "))

    heads = int(input("Nombre de têtes KV : "))
    dim = int(input("Dimension des têtes  : "))
    layers = int(input("Nombre de couches KV: "))
    quant = int(input("Quantification en octets (1, 2 ou 4) : "))
    seq = int(input("Longueur de la séquence : "))

    mon_modele = Modele(nom,editeur,archi_type)
    mon_modele.input_model_parameters(heads, dim, layers, quant)

    resultat = mon_modele.calculer_kv_cache(seq)

    print(f"==>> Pour une longeur de {seq} le modele {nom} de {editeur} necessite {resultat:.2f} Mo de KV Cache")

if __name__ == "__main__":
    main()
