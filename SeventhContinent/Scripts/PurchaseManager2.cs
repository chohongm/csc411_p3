using UnityEngine;
using System.Collections;
using UnityEngine.UI;


public class PurchaseManager2 : MonoBehaviour {

    public GameObject building;
    private BuildingManager buildingManager;

    public Text itemInfo; // text of item info to be displayed on button
    public Text itemDetail; // item detail next to button

    public Color standard; // color of button when player cannot afford
    public Color affordable; // color when can afford
    public Click click;
    public GameObject status;

    private int currency;
    private bool canAfford;
    private StatusManager statusManager;
    public int assignedWorkersCount;
    private Slider _slider;
    private bool isBuilding;
    private float timePassed;
    private string itemDetailTemp;

    private BuildingPlacement buildingPlacement;

    // Use this for initialization
    void Start() {
        // find buildingPlacement script on Main Camera.
        buildingPlacement = Camera.main.GetComponent<BuildingPlacement>();
        buildingManager = building.GetComponent<BuildingManager>();
        statusManager = status.GetComponent<StatusManager>();
        _slider = GetComponentInChildren<Slider>();
        isBuilding = false;
        timePassed = 0;
        //_slider = GetComponentInChildren<Slider>();
        // finds the first found textGO.
        itemInfo = gameObject.GetComponentsInChildren<Text>()[0];
        itemDetail = gameObject.GetComponentsInChildren<Text>()[1];
        itemInfo.text = buildingManager.GetItemInfo();
        itemDetail.text = buildingManager.GetItemDetail();

        //StatusManager = GameObject.Find(groupName).GetComponent<StatusManager>();
    }

    // Update is called once per frame
    void Update() {
        currency = statusManager.GetCurrency(buildingManager.costType);
        if (currency >= buildingManager.cost) {
            GetComponent<Image>().color = affordable;
            canAfford = true;
        } else {
            GetComponent<Image>().color = standard;
            canAfford = false;
        }
    }

    // on left click on purchase button, try to assign a worker.
    public bool CanBuild() {
        // if worker available
        if (statusManager.numIdleWorkers > 0 && canAfford) {
            return true;
        }
        return false;
    }

    // this is called when button is clicked
    public void Purchase() {
        // if current built item count is less than total buildable item count
        // and player can afford to buy a item,
        // then subtract the cost from status and increment item built count by 1.
        if (CanBuild()) {
            buildingPlacement.SetItem(building);
            /*
            statusManager.EffectMinus(buildingManager.costType, buildingManager.cost);
            isBuilding = true;
            statusManager.numIdleWorkers--;
            assignedWorkersCount++;
            // update item details for increased worker count.
            itemDetail.text = itemDetailTemp + assignedWorkersCount + " / " + buildingManager.buildWorkerCap;
            */
        }
    }
}
