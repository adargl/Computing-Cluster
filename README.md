# Cluster Computing System

<h2>Introduction</h2>
<p>This project is a Cluster Computing System that performs static code analysis on an input file written in Python. It collects information about the file and uses it to modify the abstract syntax tree (ast) created, with the aim of recognizing code structures that can be executed concurrently. The system then distributes the computation across several computers for parallel processing and collects the results to produce an output.</p>

<h2>Functionality</h2>
<p>The system collects information about the input file, including functions, loops, and threads. It then uses this information to modify the ast and recognize code structures that can be run concurrently, such as nested functions or loops. The modified code is then executed in parallel across several computers using distributed computing. Communication between the nodes is done using sockets in Python, and the required version is Python 3.8.</p>

<h2>Modules</h2>
<p>The following Python modules are used in this project:</p>
<ul>
  <li>ast</li>
  <li>astpretty</li>
  <li>pickle</li>
  <li>queue</li>
  <li>select</li>
  <li>socket</li>
  <li>time</li>
  <li>copy</li>
  <li>struct</li>
  <li>threading</li>
  <li>enum</li>
  <li>logging</li>
</ul>

<h2>Note</h2>
<p>Please note that this project is a work in progress and is the author's final project for their school studies in cyber security.</p>
