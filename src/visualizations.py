from graphviz import Digraph

def create_flowchart():
    dot = Digraph(format='png')
    dot.attr(rankdir='LR', size='10', dpi='300', fontname='Arial', fontsize='20')
    
    # Define Nodes (Structured for Clear Flow in Landscape Format)
    dot.node("B", "Capture Input\n(Voice/Text/Image)", shape='box', style='filled', fillcolor='lightblue')
    dot.node("C", "Process Image\n(BLIP-2 & LLaMA-2)", shape='box', style='filled', fillcolor='lightgreen')
    dot.node("D", "Extract Scene Insights", shape='box', style='filled', fillcolor='lightyellow')
    dot.node("E", "Get User Location", shape='box', style='filled', fillcolor='lightcoral')
    dot.node("F", "Retrieve Real-Time Info\n(Weather/Hazards)", shape='box', style='filled', fillcolor='lightpink')
    dot.node("G", "User Asks Question\n(Voice/Text)", shape='box', style='filled', fillcolor='lightsalmon')
    dot.node("H", "Generate AI Response\n(LLaMA-2)", shape='box', style='filled', fillcolor='lightgray')
    dot.node("I", "Convert Response to Audio\n(TTS)", shape='box', style='filled', fillcolor='lightsteelblue')
    dot.node("J", "Enhance Audio Clarity", shape='box', style='filled', fillcolor='plum')
    dot.node("K", "Provide Assistance\nto User", shape='box', style='filled', fillcolor='wheat')
    
    # Define Edges (Ensuring Logical Flow in Landscape)
    dot.edge("B", "C")
    dot.edge("C", "D")
    dot.edge("D", "G")
    dot.edge("B", "E")
    dot.edge("E", "F")
    dot.edge("F", "G")
    dot.edge("G", "H")
    dot.edge("H", "I")  # Moving AI response before TTS conversion
    dot.edge("I", "J")
    dot.edge("J", "K")
    
    return dot

if __name__ == "__main__":
    flowchart = create_flowchart()
    flowchart.render("visionai_flowchart1", cleanup=True)
    print("Flowchart saved as 'visionai_flowchart1.png'")
