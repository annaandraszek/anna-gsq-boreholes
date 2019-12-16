import json
import textshowing

def map_word_confidence(blocks):
    confidences = {}
    positions = {}
    i = 0
    prevPage = 0
    for block in blocks:
        if block['BlockType'] == 'WORD':
            word = block['Text']
            confidence = block['Confidence']
            if block['Page'] > prevPage:
                prevPage = block['Page']
                i = 0
            position = (block['Page'], i)
            if word not in confidences.keys():
                confidences[word] = [confidence]
                positions[word] = [position]
            else:
                confs = confidences[word]
                confs.append(confidence)
                confidences[word] = confs
                pos = positions[word]
                pos.append(position)
                positions[word] = pos
            i+=1

    #print(confidences.items())
    #print(positions.items())
    return confidences, positions

def remove_noise():
    file = 'training/fulljson/cr_24362_1_fulljson.json'
    f = json.load(open(file, "rb"))
    blocks = textshowing.json2res(f)['Blocks']
    conf_map, pos_map = map_word_confidence(blocks)
    return conf_map, pos_map

if __name__ == "__main__":
    remove_noise()