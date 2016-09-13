using UnityEngine;
using System.Collections;

public class BuildingPlacement : MonoBehaviour {

    private PlaceableBuilding placeableBuilding;
    private Transform currentBuilding;
    private BuildingManager buildingManager;
    public bool hasPlaced;
    private StatusManager statusManager;

    Ray ray;
    RaycastHit hit;
    private float raycastLength = 500;

    private Color valid;
    private Color original;
    private Color invalid;
    private GameObject tempGO;
    private MeshRenderer mr;
    private MeshRenderer[] mrs;

    // Use this for initialization
    void Start () {
        // this is so important. ClickAction keeps track of hasPalced
        // to prioritize clickable buildings when building being placed and building already placed
        // are overlapping.
        hasPlaced = true;
        statusManager = GameObject.FindWithTag("Status").GetComponent<StatusManager>();

        valid = Color.green;
        valid.a = 0.5f;
        invalid = Color.red;
        invalid.a = 0.5f;
    }
	
	// Update is called once per frame
	void Update () {
        // if building hasn't been placed yet, keep the building following cursor
	    if (currentBuilding != null && !hasPlaced) {
            ray = Camera.main.ScreenPointToRay(Input.mousePosition);
            RaycastHit[] hitGOs = Physics.RaycastAll(ray, raycastLength);
            if (hitGOs.Length > 0) {
                //Debug.Log(hit.collider.name);
                //if (hit.collider.name == "Terrain") {
                //currentBuilding.position = hit.point;
                // grid snap
                foreach (RaycastHit hit in hitGOs) {
                    if (hit.transform.tag == "HexGrid") {
                        currentBuilding.position = new Vector3(hit.transform.position.x, currentBuilding.position.y, hit.transform.position.z);
                    } 
                }
                // snap without grid
                //Debug.Log(hitGOs.Length);
                //currentBuilding.position = new Vector3(Mathf.Round(hit.point.x / 5) * 5, currentBuilding.position.y, Mathf.Round(hit.point.z / 5) * 5);
                //currentBuilding.position = hit.transform.position;
                //}
            }
            if (IsLegalPosition()) {
                ChangeToGreen();
            } else {
                ChangeToRed();
            }
            ApplyRotation();
            /*
            Vector3 m = Input.mousePosition;
            m = new Vector3(m.x, m.y, transform.position.y);
            Vector3 p = Camera.main.ScreenToWorldPoint(m);
            // terrain height is 30.
            currentBuilding.position = new Vector3(p.x, currentBuilding.position.y, p.z);
            */
        }
        // click to place the building (placing the building would subtract the cost from status)
        if (!hasPlaced && Input.GetMouseButtonDown(0)) {
            if (IsLegalPosition()) {
                ray = Camera.main.ScreenPointToRay(Input.mousePosition);
                RaycastHit[] hitGOs = Physics.RaycastAll(ray, raycastLength);
                //GameObject treeToDestory;
                foreach (RaycastHit hit in hitGOs) {
                    if (hit.transform.tag == "HexGrid") {
                        BuildingPlaced(hit.transform);
                        Camera.main.GetComponent<ClickAction>().enabled = true;
                    }
                }
                // turn off grids
                foreach (Transform hex in GameObject.Find("HexGrid").transform) {
                    hex.GetComponent<MeshRenderer>().enabled = false;
                }
            }
            // right click to cancel building placement.
        } else if (!hasPlaced && Input.GetMouseButtonDown(1)) {
            UnsetVars();
            Destroy(currentBuilding.gameObject);
            foreach (Transform hg in GameObject.Find("HexGrid").transform) {
                hg.GetComponent<MeshRenderer>().enabled = false;
            }
        }
    }

    bool IsLegalPosition() {
        if (placeableBuilding.colliders.Count > 0) {
            Debug.Log("Illegal Position!");
            return false;
        }
        return true;
    }

    void ChangeToGreen() {
        if (mr) {
            currentBuilding.gameObject.GetComponent<MeshRenderer>().material.color = valid;
        } else {
            for (int i = 0; i < mrs.Length; i++) {
                if (mrs[i].materials.Length > 1) {
                    for (int j = 0; j < mrs[i].materials.Length; j++) {
                        mrs[i].materials[j].color = valid;
                    }
                } else {
                    mrs[i].material.color = valid;
                }
            }
        }
    }

    /* // not working
    void ChangeToOriginalColor() {
        MeshRenderer[] mrsOriginal = tempGO.transform.GetChild(0).GetComponentsInChildren<MeshRenderer>();
        for (int i = 0; i < mrs.Length; i++) {
            if (mrs[i].materials.Length > 1) {
                for (int j = 0; j < mrs[i].materials.Length; j++) {
                    mrs[i].materials[j].color = mrsOriginal[i].materials[j].color;
                }
            } else {
                mrs[i].material.color = mrsOriginal[i].material.color;
            }
        }
    }
    */

    void ChangeToRed() {
        if (mr) {
            currentBuilding.gameObject.GetComponent<MeshRenderer>().material.color = invalid;
        } else {
            for (int i = 0; i < mrs.Length; i++) {
                if (mrs[i].materials.Length > 1) {
                    for (int j = 0; j < mrs[i].materials.Length; j++) {
                        mrs[i].materials[j].color = invalid;
                    }
                } else {
                    mrs[i].material.color = invalid;
                }
            }
        }
    }

    // called when building purchase button is clicked
    public void SetItem(GameObject b) {
        tempGO = b;
        currentBuilding = ((GameObject)Instantiate(b)).transform;
        if (currentBuilding.gameObject.GetComponent<MeshRenderer>()) {
            mr = currentBuilding.gameObject.GetComponent<MeshRenderer>();
        } else {
            mrs = currentBuilding.GetChild(0).GetComponentsInChildren<MeshRenderer>();
        }
        ChangeToGreen();
        buildingManager = currentBuilding.GetComponent<BuildingManager>();
        placeableBuilding = currentBuilding.GetComponent<PlaceableBuilding>();
        hasPlaced = false;
        // display hexgrid
        foreach (Transform hg in GameObject.Find("HexGrid").transform) {
            hg.GetComponent<MeshRenderer>().enabled = true;
        }
        Camera.main.GetComponent<ClickAction>().enabled = false;
    }

    public void UnsetVars() {
        // update no longer changes position of the building object.
        hasPlaced = true;
        mr = null;
        mrs = null;
        tempGO = null;
        buildingManager = null;
        placeableBuilding = null;
    }

    void replaceNewPrefab() {
        Destroy(currentBuilding.gameObject);
        currentBuilding = ((GameObject)Instantiate(tempGO)).transform;
        buildingManager = currentBuilding.GetComponent<BuildingManager>();
        buildingManager.Init();
        placeableBuilding = currentBuilding.GetComponent<PlaceableBuilding>();
    }

    public void ResetBuildingMesh() {
        if (buildingManager.buildingType == "House" || buildingManager.buildingType == "Statue") {
            // replace current building with original prefab
            Vector3 tempPos = currentBuilding.position;
            Vector3 tempAngle = currentBuilding.eulerAngles;
            replaceNewPrefab();
            currentBuilding.position = tempPos;
            currentBuilding.eulerAngles = tempAngle;
        } else {
            if (currentBuilding.gameObject.GetComponent<MeshRenderer>()) {
                currentBuilding.gameObject.GetComponent<MeshRenderer>().material.color = Color.white;
            } else {
                for (int i = 0; i < mrs.Length; i++) {
                    mrs[i].material.color = Color.white;
                }
            }
        }
    }

    // resets building to normal color and passes grid trees to be destroyed when build process is compelte.
    void BuildingPlaced(Transform hex) {
        ResetBuildingMesh();
        currentBuilding.parent = hex;
        // activate HexManager for activating mouseover() for worker assignment to its child building
        //hex.GetComponent<HexManager>().enabled = true;

        // subtract the cost from status
        statusManager.EffectMinus(buildingManager.costType, buildingManager.cost);
        // built complete now clicking makes it worker assignable.
        //currentBuilding.GetComponent<BuildingManager>().build();
        // set hasPlaced in BM to true to activate onMouseOver for clicks
        // has to be after above.
        buildingManager.AssignWorker();
        // activate GUI
        buildingManager.GUI.gameObject.SetActive(true);
        buildingManager.hasPlaced = true;
        buildingManager.treeToDestroy = hex.GetChild(0).gameObject;
        UnsetVars();
    }

    // scroll to rotate building to be placed.
    void ApplyRotation() {
        float deadZone = 0.01f;
        float ease = 3500f;
        float scrollWheelValue = Input.GetAxis("Mouse ScrollWheel") * ease * Time.deltaTime;

        if ((scrollWheelValue > -deadZone && scrollWheelValue < deadZone) || scrollWheelValue == 0f)
            return;

        if (scrollWheelValue > 0) {
            scrollWheelValue = 60;
        } else {
            scrollWheelValue = -60;
        }
        Debug.Log(scrollWheelValue);

        currentBuilding.eulerAngles = new Vector3(currentBuilding.eulerAngles.x, currentBuilding.eulerAngles.y + scrollWheelValue, currentBuilding.eulerAngles.z);
    }
}
