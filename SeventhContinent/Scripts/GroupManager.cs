using UnityEngine;
using System.Collections;

// this script is attached to every purchasable object groups
// used for activating their child objects when purchased.
public class GroupManager : MonoBehaviour {

    public GameObject button;
	// Use this for initialization
	void Start () {
        foreach (Transform child in transform) {
            child.gameObject.SetActive(false);
        }
    }
	
	// houseBuilt count goes up as more house is purchased
    /*
	void Update () {
        houseIndex = button.GetComponent<PurchaseManager>().houseBuilt - 1;
        if (houseIndex >= 0 && !transform.GetChild(houseIndex).gameObject.activeSelf) {
            transform.GetChild(houseIndex).gameObject.SetActive(true);
        }
	}
    */

    public void Build(int buildingInd) {
        //houseIndex = button.GetComponent<PurchaseManager>().houseBuilt - 1;
        if (buildingInd >= 0 && !transform.GetChild(buildingInd).gameObject.activeSelf) {
            transform.GetChild(buildingInd).gameObject.SetActive(true);
        }
    }

    public GameObject GetChildBuilding(int buildingInd) {
        if (buildingInd >= 0 && !transform.GetChild(buildingInd).gameObject.activeSelf) {
            return transform.GetChild(buildingInd).gameObject;
        }
        return null;
    }
}
