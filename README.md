<!DOCTYPE html>
<html>
<head>
  <title>Cluster Computing System</title>
</head>
<body>
  <h1>Cluster Computing System</h1>

  <h2>Introduction</h2>
  <p>This project is a Cluster Computing System that performs static code analysis on an input file written in Python. It collects information about the file and uses it to modify the abstract syntax tree (AST) created, with the aim of recognizing code structures that can be executed concurrently. The system then distributes the computation across several computers for parallel processing and collects the results to produce an output.</p>

  <h2>Functionality</h2>
  <p>The system collects information about the input file, including functions, loops, and threads. It then uses this information to modify the AST and recognize code structures that can be run concurrently, such as nested functions or loops. The modified code is then executed in parallel across several computers using distributed computing. Communication between the nodes is done using sockets in Python, and the required python version is Python 3.9.</p>

  <h2>Installation</h2>
  <ol>
    <li>Download the package <code>PyQt6-Qscintilla</code> from the official website and install it on your system.</li>
    <li>Download the package <code>pyenchant</code> and install it using your preferred package manager (e.g., pip or conda).</li>
  </ol>

  <h2>Running the Project</h2>
  <p>Follow these steps to run the project:</p>
  <ol>
    <li>Open a terminal or command prompt.</li>
    <li>Navigate to the project directory.</li>
    <li>Run <code>Manager.py</code> by executing the following command: <code>python Manager.py</code></li>
    <li>After <code>Manager.py</code> has started, open a new terminal or command prompt.</li>
    <li>Navigate to the project directory.</li>
    <li>Run <code>Node.py</code> by executing the following command: <code>python Node.py</code></li>
    <li>Finally, Navigate to the project directory.</li>
    <li>Finally, run <code>GUI/main.py</code> by executing the following command: <code>python GUI/main.py</code> and a GUI will appear</li>
  </ol>

  <p>Additional steps are required when running Node.py or GUI/main.py on a different computer other than the one where Manager.py (=Server) is executed. For example, to add nodes to the cluster, follow these steps:</p>
  <ol>  
    <li>open the <code>Node.py</code> file in a text editor.</li>
    <li>Scroll to the bottom of the file and locate the line: <code>client = Node("localhost", 10)</code></li>
    <li>Replace <code>"localhost"</code> with the correct IP address of the server where Manager.py is running.</li>
    <li>Optionally, adjust the second parameter (10 in this example) to specify the number of threads a node will execute tasks on.</li>
    <li>Save the changes to <code>Node.py</code> and execute the files as previously.</li>
  </ol>

  <p> Alternatively, in order to run the GUI from a different computer, follow these steps:</p>
  <ol>
    <li>open the <code>main.py</code> file in a text editor.</li>
    <li>Scroll to the bottom of the file and locate the line: <code>window = MainWindow("localhost")</code></li>
    <li>Replace <code>"localhost"</code> with the correct IP address of the server where Manager.py is running.</li>
    <li>Save the changes to <code>main.py</code> and execute the files as previously.</li>
  </ol>


  <h2>Note</h2>
  <p>This project is the author's final project for their school studies in cyber security.</p>
</body>
</html>