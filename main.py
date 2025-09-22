

class modele:
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
        - quantization_oct: 2 for bf16, 1 for 8bits
        """
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
        
        # step 2: De la brique à la séquence entière
        # Chaque bloc contient 1 couche d'attention, le bloc est répété N fois (nbr_Gated_Attention_layers)
        elements=elmt_per_token_per_layer*length_seq*(self.nbr_Gated_Attention_layers)

        # step 3 : Des éléments aux octets
        kv_cache_oct=elements*self.quantization_oct
        kv_cache=kv_cache_oct/(2**20)

        return kv_cache





def main():
    print("Hello from rtx-llm-engine!")


if __name__ == "__main__":
    main()
