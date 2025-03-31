import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  # Use TensorFlow 1.x behavior for loading `.pb`

def load_graph(model_path):
    # Load the protobuf file from the disk
    with tf.io.gfile.GFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    
    # Import the graph into the current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    return graph

# Specify your `.pb` file path
model_path = r"C:\Users\kbthe\Downloads\Masked-Face-Recognition-main\weights_15.pb"
graph = load_graph(model_path)

# Print operations in the graph to verify the loading
for op in graph.get_operations():
    print(op.name)
