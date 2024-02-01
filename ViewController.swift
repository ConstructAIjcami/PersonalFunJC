//
//  ViewController.swift
//  ConstructionAR 1
//
//  Created by Jasmin Cami on 1/29/20.
//  Copyright © 2020 Construct AI, Inc. All rights reserved.
//

import UIKit
import RealityKit

class ViewController: UIViewController {
    
    @IBOutlet var arView: ARView!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Load the "Box" scene from the "Experience" Reality File
        let boxAnchor = try! Experience.loadBox()
        
        // Add the box anchor to the scene
        arView.scene.anchors.append(boxAnchor)
      
    }
}
