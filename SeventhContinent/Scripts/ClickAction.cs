using UnityEngine;
using System.Collections;

public class ClickAction : MonoBehaviour {

    private bool didHit;
    private Ray ray;
    private RaycastHit rhInfo;
    private BuildingPlacement bp;
    private RaycastHit[] hitGOs;
    private int raycastLength;

    void Start() {
        bp = Camera.main.GetComponent<BuildingPlacement>();
        this.enabled = false;
    }

    // Update is called once per frame
    void Update () {
        // If this value is true, you should be hovering any of your UI elements, if false not.
        // make sure you also uncheck the 'Blocks Raycast' when you hide it. (otherise Unity will tell yo that you are hovering the invisible UI)
        if (!UnityEngine.EventSystems.EventSystem.current.IsPointerOverGameObject()) {
            ray = Camera.main.ScreenPointToRay(Input.mousePosition);
            // hasPlaced means every building is placed and no building is BEING placed.
            // therefore, every building on the screen can be interacted.

            if (bp.hasPlaced) {
                didHit = Physics.Raycast(ray, out rhInfo);
                if (didHit) {
                    if (rhInfo.transform.tag == "EffectBuilding" || rhInfo.transform.tag == "ProductionBuilding") {
                        rhInfo.transform.GetComponent<BuildingManager>().MouseOver();
                    }
                }
            }
        }
        /*
        if (bp.hasPlaced) {
            didHit = Physics.Raycast(ray, out rhInfo);
            Debug.Log(rhInfo.transform.name);
            if (didHit && rhInfo.transform.tag == "Building") {
                rhInfo.transform.GetComponent<BuildingManager>().MouseOver();
            } else if (didHit && rhInfo.transform.GetChild(0).tag == "Building") {
                rhInfo.transform.GetChild(0).GetComponent<BuildingManager>().MouseOver();
            }
        }

       ray = Camera.main.ScreenPointToRay(Input.mousePosition);
        // hasPlaced means every building is placed and no building is BEING placed.
        // therefore, every building on the screen now can be interacted.
        if (bp.hasPlaced) {
            hitGOs = Physics.RaycastAll(ray, raycastLength);
            if (hitGOs.Length > 0) {
                foreach (RaycastHit hit in hitGOs) {
                    if (hit.transform.tag == "HexGrid") {
                        hit.transform.GetComponent<BuildingManager>().MouseOver();
                    }
                }
            }
        }
        */
    }
}
