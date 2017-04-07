(ns clj-tf.examples
    ^{:author "Peter Feldtmann"
      :doc "Usage examples for clj-tf.
            Use this as blueprints for your own stuff."}
  (:require [clj-tf.core :as tf]))

;;---------------------------------------------------------------------------

(defn print-version
  "Very simple first test:
  Get the version of the underlying TensorFlow implementation."
  []
  (println "Using TensorFlow version" (tf/get-version)))


;;---------------------------------------------------------------------------

(defn hello-world
  "Hello World example from web page
   'Installing TensorFlow for Java: Validate the installation'."
  []
  (tf/with-new-graph g
    (let [msg 
          (str "Hello from TensorFlow version " (tf/get-version) "!")
          
          ;; Construct the computation graph with a single operation, a constant
          ;; named "MyConst" with a value "msg".
          value (tf/make-tensor-from-string msg)
          
          _ (tf/const-tensor g "MyConst" value)
          
          ;; Execute the "MyConst" operation in a Session.
          result
          (tf/run-and-process g :fetches ["MyConst"])
          ]
      (println (tf/->string result))
      (tf/destroy result))))

;;---------------------------------------------------------------------------

(declare execute-inception-graph 
         construct-and-execute-graph-to-normalize-image)

(defn label-image
  "Example adapted from the original java examples at
      <tensorflow-project>/java/src/main/java/org/tensorflow/examples/LabelImage.java 
   which demonstrates a sample use of the TensorFlow Java API
   to label images using a pre-trained model (http://arxiv.org/abs/1512.00567).
   Download the pre-trained inception model from
      https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
   and unzip it into folder 'resources/inception'.
   <image-file> is the path and name to the JPEG image file you want to label."
  [image-file]
  (let [graph-def (tf/read-all-bytes "resources/inception/tensorflow_inception_graph.pb")
        labels (tf/read-all-lines "resources/inception/imagenet_comp_graph_label_strings.txt")
        image-bytes (tf/read-all-bytes image-file)]
    (tf/with-tensor image
      (construct-and-execute-graph-to-normalize-image image-bytes)
      (let [label-probabilities
            (execute-inception-graph graph-def image)
            
            best-label-idx
            (tf/max-index-of-list (tf/array-as-list label-probabilities))
            
            best-label
            (get labels best-label-idx)
            ]
        (println (str "Best match: '" best-label
                      "', likelyhood: "
                      (* (get label-probabilities best-label-idx) 100) " %"))
        best-label))))
 
(defn- execute-inception-graph
  [graph-def image-tensor]
  (tf/with-new-graph g 
    (tf/without-root-scope
      (tf/import-graph-def g graph-def)
      #_(tf/run-and-process-simple-session g 
                                          "output"
                                          "input" image-tensor
                                          #(first (tf/->floats (first %))))
      (tf/run-and-process g 
              :fetches ["output"]
              :feeds [["input" image-tensor]]
              :proc-fn #(first (tf/->floats (first %)))))))
 
(defn- construct-and-execute-graph-to-normalize-image
  [imageBytes]
  ;;(println (str imageBytes) (type imageBytes))
  (tf/with-root-scope "construct-and-normalize"
    (tf/with-new-graph g
;      Some constants specific to the pre-trained model at:
;       https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
;       - The model was trained with images scaled to 224x224 pixels.
;       - The colors, represented as R, G, B in 1-byte each were converted to
;         float using (value - Mean)/Scale.
    (let [h 224
          w 224
          mean (float 117)
          scale (float 1)
          
;         Since the graph is being constructed once per execution here, we can use a constant for the
;         input image. If the graph were to be re-used for multiple input images, a placeholder would
;         have been more appropriate.
          input 
          (tf/constant g "input" imageBytes)
          
          output
          (tf/div g 
              (tf/sub g
                  (tf/resize-bilinear g
                      (tf/expand-dims g
                          (tf/cast g
                             (tf/decode-jpeg g input (long 3))
                             (tf/dtype :float))
                          (tf/constant g "make_batch" (int 0)))
                      (tf/constant g "size", (int-array [h w])))                  
                  (tf/constant g "mean" mean))
              (tf/constant g "scale" scale))
          ]
      #_(tf/run-and-process-simple-session g
                                         (-> output tf/get-output-op tf/get-op-name)
                                         nil nil first)
      (tf/run-and-process g 
            :fetches [(-> output tf/get-output-op
                        tf/get-raw-op-name)])) )))

;;---------------------------------------------------------------------------
;; run the examples
;;---------------------------------------------------------------------------

(comment
  
  (print-version)
  
  (hello-world)
  
  ;; is it a mushroom? Yes!
  (label-image "resources/inception/agaric.jpg")
  
  ;; funny: Best match is "crossword puzzle", but in a way close indeed ;-)
  (label-image "resources/inception/chessboard.jpg")
  
  (tf/get-all-op-names)
  
  ;; misc. experiments, ignore this:
  
  (use 'flatland.protobuf.core)
  (import org.tensorflow.framework.OpDef)
  (def x (protodef org.tensorflow.framework.OpDef))
  (def s (protobuf-schema x))
  (def x1 (protobuf x :name "Bob" )) ;; map
  (def x1dump (protobuf-dump x1)) ;; byte-array
  (def y (protobuf-load x x1dump)) ;; map
    
  (def src (slurp "resources/ops.txt"))
  
  (import org.tensorflow.framework.OpList)
  (def l (protodef org.tensorflow.framework.OpList))
  (def ls (protobuf-schema l))
   
  (def textparser (com.google.protobuf.TextFormat/getParser))
  (def builder (OpList/newBuilder))
  (.merge textparser src builder)
  (def m (hash-map (first (.getAllFields builder))))
  
  ;;(def tp (com.google.protobuf.TextFormat$Parser/newBuilder))
  (com.google.protobuf.TextFormat/merge src builder)
  (def b (.build builder))
  (def op (.getOp builder 0));; Opdef
  (.getName op)
  (.getAttrList op)
  (.getSummary op)
  (.getInputArgList op)
  (.getName (.getInputArg op 0))
  
  (.getOpList builder)
  
  )
