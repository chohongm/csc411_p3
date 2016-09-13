using UnityEngine;
using System.Collections;


public class RightClick : MonoBehaviour{

    public bool isRightClick;
	// Use this for initialization
	void Start () {
	}
	
	// Update is called once per frame
	void Update () {
	    if (Input.GetMouseButtonUp(0)) {
            isRightClick = false;
        } else if (Input.GetMouseButtonUp(1)) {
            isRightClick = true;
        }
	}

    void CheckRightClick() {
        if (isRightClick) {

        }
    }
}
