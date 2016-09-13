using UnityEngine;
using System.Collections;

public class HexManager : MonoBehaviour {

	// Use this for initialization
	void Start () {
        this.enabled = false;
	}
	
	// Update is called once per frame
	void Update () {
	
	}

    // first click display +- buttons on canvas.
    // + button click > assign worker
    // - button click > removeworker
    // 2nd click on building or clicking on other building turns off canvas
    public void MouseOver() {
        // if building is assignable and complete,
        Transform child = transform.GetChild(0);
        if (child && child.tag == "Building") {
            BuildingManager bm = child.gameObject.GetComponent<BuildingManager>();
            // left click, increment
            if (Input.GetMouseButtonDown(0)) {
                bm.AssignWorker();
            }
            if (Input.GetMouseButtonDown(1)) {
                // right click, decrement
                bm.RemoveWorker();
            }
            if (Input.GetMouseButtonDown(2)) {
                // middle click, increment all
                bm.AssignAll();
            }
        }
    }
}
