using UnityEngine;
using System.Collections;
using UnityEngine.UI;

public class PurchaseManager : MonoBehaviour {

    public int numBuilt;
    public string groupName;
    public string itemName;
    public int cost;
    public string costType;
    public float duration;
    public string effectType;
    public int effectValue;
    public int effectPerSeconds;
    public int maxWorkersCount;
    public Text itemInfo; // text of item info to be displayed on button
    public Text itemDetail; // item detail next to button

    public Color standard; // color of button when player cannot afford
    public Color affordable; // color when can afford
    public Click click;
    public GameObject status;

    private int totNumGOs;
    private int currency;
    private bool canAfford;
    private GameObject groupGO;
    private GroupManager groupManager;
    private StatusManager statusManager;
    public int assignedWorkersCount;
    private Slider _slider;
    private bool isBuilding;
    private float timePassed;
    private string itemDetailTemp;

    private BuildingPlacement buildingPlacement;

    // Use this for initialization
    void Start () {
        // find buildingPlacement script on Main Camera.
        buildingPlacement = Camera.main.GetComponent<BuildingPlacement>();
        numBuilt = 0;
        groupGO = GameObject.Find(groupName);
        totNumGOs = groupGO.transform.childCount;
        groupManager = groupGO.GetComponent<GroupManager>();
        statusManager = status.GetComponent<StatusManager>();
        _slider = GetComponentInChildren<Slider>();
        isBuilding = false;
        timePassed = 0;

        //_slider = GetComponentInChildren<Slider>();
        // finds the first found textGO.
        itemInfo = gameObject.GetComponentsInChildren<Text>()[0];
        itemDetail = gameObject.GetComponentsInChildren<Text>()[1];
        itemInfo.text = itemName + "\nCost: " + cost + " " + costType;
        itemDetailTemp = "Duration : " + duration / 60 + " min\n" + effectType + " +" + effectValue;
        switch (itemName) {
            case "Farm":
                itemDetailTemp += " / worker / " + effectPerSeconds + " seconds";
                break;
            case "Lumber Mill":
                itemDetailTemp += " / worker / " + effectPerSeconds + " seconds";
                break;
            case "Stone Work":
                itemDetailTemp += " / worker / " + effectPerSeconds + " seconds";
                break;
            default:
                break;
        }
        //"Time Left : " + (duration - timePassed) / 60 + " min\n" +
        itemDetailTemp += "\nWorkers Assigned : ";
        itemDetail.text = itemDetailTemp + assignedWorkersCount + " / " + maxWorkersCount;
        //StatusManager = GameObject.Find(groupName).GetComponent<StatusManager>();
    }
	
	// Update is called once per frame
	void Update () {
        currency = statusManager.GetCurrency(costType);
        if (currency >= cost && numBuilt < totNumGOs) {
            GetComponent<Image>().color = affordable;
            canAfford = true;
        } else {
            GetComponent<Image>().color = standard;
            canAfford = false;
        }

        if (isBuilding) {
            _slider.value = (timePassed / duration) * 100;
            // build is complete. reset slider, isBuilding to false, assigned workers to 0.
            if (timePassed >= duration) {
                buildComplete();
            // build in process.
            } else {
                // each assigned worker increase building process by 5%
                timePassed += (((assignedWorkersCount * 0.1f) + 1) * Time.deltaTime);
            }
        }
    }

    // on left click on purchase button, try to assign a worker.
    public bool CanAssignWorker() {
        // if worker available and assigned worker count is not at max, assign worker.
        if (statusManager.numIdleWorkers > 0 && assignedWorkersCount < maxWorkersCount) {
            return true;
        }
        return false;
    }

    // this is called when button is clicked
    public void Purchase() {
        // if current built item count is less than total buildable item count
        // and player can afford to buy a item,
        // then subtract the cost from status and increment item built count by 1.
        if (CanAssignWorker()) {
            // if already building, just assign worker.
            if (isBuilding) {
                statusManager.numIdleWorkers--;
                assignedWorkersCount++;
                // update item details for increased worker count.
                itemDetail.text = itemDetailTemp + assignedWorkersCount + " / " + maxWorkersCount;
            // if just purchased, start building and assign worker.
            } else if (numBuilt < totNumGOs && canAfford) {
                buildingPlacement.SetItem(groupManager.GetChildBuilding(numBuilt));
                statusManager.EffectMinus(costType, cost);
                isBuilding = true;
                statusManager.numIdleWorkers--;
                assignedWorkersCount++;
                // update item details for increased worker count.
                itemDetail.text = itemDetailTemp + assignedWorkersCount + " / " + maxWorkersCount;
            }
        }
    }
    // called when purchase button slide reaches 100 (when building completes).
    public void buildComplete() {
        // build house at index and increment built count.
        groupManager.Build(numBuilt);
        // increase the effect type by its value on status
        if (effectType == "Population" || effectType == "Fame") {
            statusManager.EffectPlus(effectType, effectValue);
        }
        numBuilt += 1;
        // free all assigned workers.
        statusManager.numIdleWorkers += assignedWorkersCount;
        assignedWorkersCount = 0;
        itemDetail.text = itemDetailTemp + assignedWorkersCount + " / " + maxWorkersCount;
        // reset build status
        isBuilding = false;
        timePassed = 0;
        _slider.value = 0;
    }
}
