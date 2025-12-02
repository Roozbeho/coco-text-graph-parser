from factual_scene_graph.parser.scene_graph_parser import SceneGraphParser
from typing import Any
import json
import argparse
from tqdm import tqdm

class T5TextGraphParse:
    def __init__(
        self,
        model_name: str = "lizhuang144/flan-t5-base-VG-factual-sg-id",
        parser_type: str = 'DiscoSG-Refiner',
        refiner_checkpoint_path: str = 'sqlinn/DiscoSG-Refiner-Large-t5-only',
        task: str = "delete_before_insert",          
        device: str = "cuda",        
        refinement_rounds: int = 3,
        batch_size: int = 32,
        beam_size: int = 9,
        max_input_len=4096,
        max_output_len=4096,
        merge_strategy: str = "sentence_merge",  
        **kwargs,
    ):
        self.parser = SceneGraphParser(
            model_name,
            parser_type=parser_type,
            refiner_checkpoint_path=refiner_checkpoint_path,
            device=device
        )
        self.task = task
        self.refinement_rounds = refinement_rounds
        self.batch_size = batch_size
        self.merge_strategy = merge_strategy
        self.beam_size = beam_size
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def _convert(self, graph: dict[str, Any]) -> list[dict[str, Any]]:
        id_counter = 0
        idx_mapping = {}
        

        def get_pk(name: str):
            nonlocal id_counter, idx_mapping
            if name not in idx_mapping:
                idx_mapping[name] = id_counter
                id_counter += 1
            return idx_mapping[name]
        
        # preprocess
        for subj_idx, ent in enumerate(graph['entities']):
            get_pk(ent['head'])
        
        triplets = []
        for subj_idx, ent in enumerate(graph['entities']):
            head = ent['head']
            head_idx = get_pk(head)
            
            for attr in ent['attributes']:
                attr_id = get_pk(attr)
                
                triplets.append({
                    'subject': ent['head'],
                    'predicate': 'is',
                    'object': attr,
                    'subject_idx': head_idx,
                    'object_idx': attr_id
                })
        
        for rel in graph['relations']:
            subj = graph['entities'][rel['subject']]['head']
            obj = graph['entities'][rel['object']]['head']
            pred = rel['relation'][rel['relation'].index(':')+1:] if ':' in rel['relation'] else rel['relation']
            triplets.append({
                'subject': subj,
                'predicate': pred,
                'object': obj,
                'subject_idx': get_pk(subj),
                'object_idx': get_pk(obj)
            })
        
        return triplets
    
    def parse_captions(self, captions: list[str]) -> list[list[dict[str, Any]]]:
        results = []
        graphs = self.parser.parse(captions, beam_size=self.beam_size, refinement_rounds=self.refinement_rounds,
                             task=self.task, batch_size=self.batch_size, return_text=False,
                             max_input_len=self.max_input_len, max_output_len=self.max_output_len)
        
        # print(graphs)
        for g in graphs:
            results.append(self._convert(g))
        
        return results
    
parser = argparse.ArgumentParser(description="Parse captions into scene graphs")
parser.add_argument(
    "--caption_file_path", type=str, required=True,
    help="coco caption json file"
) 
parser.add_argument("--batch_size", type=int, default=32, required=False, help="batch size")
parser.add_argument("--max_input_len", type=int, default=4096, required=False, help="max_input_len")
parser.add_argument("--max_output_len", type=int, default=4096, required=False, help="max_output_len")
args = parser.parse_args()


with open(args.caption_file_path, 'r') as f:
    file = json.load(f)
    

graph_parser = T5TextGraphParse(max_input_len=args.max_input_len, max_output_len=args.max_output_len)

BATCH_SIZE = args.batch_size      
CHUNK_SAVE = 100000    

res = {}
counter = 0

annotations = file['annotations']

for batch_start in tqdm(range(0, len(annotations), BATCH_SIZE), desc="Processing captions"):
    batch = annotations[batch_start : batch_start + BATCH_SIZE]

    captions = [ann['caption'] for ann in batch]

    batch_graphs = graph_parser.parse_captions(captions)

    for ann, graph in zip(batch, batch_graphs):
        ids = ann['id']
        image_id = ann['image_id']
        caption = ann['caption']

        res[ids] = {
                'image_id': image_id,
                'caption': caption,
                'graph': graph
            }

    if len(res) >= CHUNK_SAVE:
        print('saved ', counter)
        with open(f'/data/comp-vlm-data/coco/text_graphs_files/text_graphs_{counter}.json', 'w') as f:
            json.dump(res, f)
        counter += 1
        res = {}

if res:
    with open(f'/data/comp-vlm-data/coco/text_graphs_files/text_graphs_{counter}.json', 'w') as f:
        json.dump(res, f)
